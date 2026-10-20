# SPDX-License-Identifier: Apache-2.0

import functools
import math
from contextlib import contextmanager
from typing import Any

import torch

from areal.api import TrainEngine
from areal.api.cli_args import MOPDLossConfig, PPOActorConfig, RejectionSamplingConfig
from areal.engine.core import stage_batch_for_engine
from areal.engine.r3.transport import (
    clear_engine_r3_side_channel,
    localize_r3_tensor,
    pop_r3_tensors,
    set_engine_r3_side_channel,
)
from areal.infra import TrainController
from areal.infra.rpc.serialization import serialize_value
from areal.trainer.mopd.loss import compose_mopd_loss
from areal.trainer.mopd.targets import aggregate_mopd_targets
from areal.trainer.ppo.advantage import (
    AdvantageInputs,
    _get_truncated_mask,
    resolve_advantage_fn,
    validate_advantage_result,
)
from areal.trainer.ppo.advantage_algorithms import (
    _shape_advantages_with_gvpo as _shape_advantages_with_gvpo,
)
from areal.trainer.ppo.advantage_algorithms import (
    _shape_advantages_with_process_weighting as _shape_advantages_with_process_weighting,
)
from areal.trainer.ppo.loss_reduction import (
    PreparedLossStep,
    prepare_policy_gradient_batch,
)
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils import logging, stats_tracker
from areal.utils.constants import (
    PROX_APPROX_METHOD_LINEAR,
    PROX_APPROX_METHOD_LOGLINEAR,
    PROX_APPROX_METHOD_ROLLOUT,
    PROX_APPROX_METHODS_ALL,
    PROX_LOGP_METHOD_LOGLINEAR,
    PROX_LOGP_METHOD_METRICS,
    PROX_LOGP_METHOD_RECOMPUTE,
    ProxLogpMethod,
)
from areal.utils.data import (
    TrajBatchMeta,
    batched_call,
    concat_batch,
    is_multi_modal_key,
    split_training_batch_into_microbatches,
)
from areal.utils.functional import (
    apply_rejection_sampling,
    cispo_loss_fn,
    ppo_actor_loss_fn,
    sapo_loss_fn,
)
from areal.utils.functional.loss_aggregation import (
    LossAggregationMode,
    TokenMean,
)
from areal.utils.perf_tracer import trace_perf
from areal.utils.stats_tracker import ReduceType
from areal.v2.training_service.controller.controller import (
    GatewayTrainController,
)

logger = logging.getLogger("PPOActor")


def _group_training_metrics(
    loss_mask: torch.Tensor,
    group_sizes: list[int],
    logical_group_sizes: list[int] | None = None,
    loss_aggregation: LossAggregationMode = "token_mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = loss_mask.shape[0]
    if any(size < 1 for size in group_sizes) or sum(group_sizes) != batch_size:
        raise ValueError(
            f"group_sizes must be positive and sum to batch size {batch_size}, "
            f"got {group_sizes}"
        )

    group_starts = torch.zeros(batch_size, dtype=torch.bool, device=loss_mask.device)
    usable_group_sizes = torch.zeros(
        batch_size, dtype=torch.float32, device=loss_mask.device
    )
    group_loss_weights = torch.zeros_like(usable_group_sizes)
    sizes = torch.tensor(group_sizes, dtype=torch.long, device=loss_mask.device)
    ends = sizes.cumsum(0)
    starts = ends - sizes
    token_counts = loss_mask.reshape(batch_size, -1).sum(1, dtype=torch.float32)
    unit_counts = (
        token_counts if loss_aggregation == "token_mean" else (token_counts > 0).float()
    )
    cumulative_units = torch.nn.functional.pad(unit_counts.cumsum(0), (1, 0))

    group_starts[starts] = True
    usable_group_sizes[starts] = torch.tensor(
        logical_group_sizes if logical_group_sizes is not None else group_sizes,
        dtype=usable_group_sizes.dtype,
        device=loss_mask.device,
    )
    weights = cumulative_units[ends] - cumulative_units[starts]
    if loss_aggregation == "prompt_mean":
        weights = (weights > 0).float()
    group_loss_weights[starts] = weights
    return group_starts, usable_group_sizes, group_loss_weights


def _infer_prompt_lens(
    attention_mask: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    """Return the index of the first generated token for each trajectory.

    ``loss_mask`` arrives rolled left by one (see ``_compute_advantages``), so it
    marks the position that *predicts* each generated token. Undo the roll before
    locating the first one, otherwise every prompt length comes out one short.
    """
    loss_mask_long = torch.roll(loss_mask.long(), shifts=1, dims=-1)
    first_gen_idx = loss_mask_long.argmax(dim=-1)
    has_gen = loss_mask_long.any(dim=-1)
    return torch.where(has_gen, first_gen_idx, attention_mask.long().sum(-1))


class PPOActor:
    def __init__(self, config: PPOActorConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine

        self.advantage_fn = resolve_advantage_fn(config)

        self.temperature = config.temperature

        self.m2_threshold = config.m2_threshold
        self._mopd_loss_config: MOPDLossConfig | None = None

        # Log critical GSPO/GRPO configuration for reproducibility
        self._log_configuration()

    def configure_mopd_loss(self, config: MOPDLossConfig) -> None:
        """Bind static MOPD loss settings once on each actor worker."""
        if (
            config.distillation_coefficient != 0
            and self.config.loss_aggregation != "token_mean"
        ):
            raise ValueError(
                "MOPD distillation is only supported with "
                "loss_aggregation='token_mean'."
            )
        if self._mopd_loss_config is not None and self._mopd_loss_config != config:
            raise RuntimeError("MOPD loss configuration is already bound")
        self._mopd_loss_config = config

    def _log_configuration(self):
        """Log PPO configuration including how proximal policy is computed."""
        config = self.config

        logger.info("=" * 70)
        logger.info("PPOActor Configuration")
        logger.info("=" * 70)

        # Log PPO mode and proximal policy computation
        if not config.use_decoupled_loss:
            logger.info("Mode: Standard PPO (on-policy)")
            if config.recompute_logprob:
                logger.info("  old_logp (π_old): RECOMPUTED from current policy")
            else:
                logger.info(
                    "  old_logp (π_old): FROM INFERENCE (cached during rollout)"
                )
        else:
            logger.info("Mode: Decoupled PPO (off-policy)")
            logger.info("  log_p_behave (π_behave): FROM INFERENCE (behavior policy)")

            # Log proximal policy computation method
            method_descriptions = {
                PROX_LOGP_METHOD_RECOMPUTE: "RECOMPUTED via forward pass (standard decoupled PPO)",
                PROX_LOGP_METHOD_LOGLINEAR: "LOG-LINEAR APPROXIMATION (no forward pass)",
                PROX_LOGP_METHOD_METRICS: "RECOMPUTED + APPROXIMATION METRICS (for evaluation)",
            }
            desc = method_descriptions.get(
                config.prox_logp_method, f"UNKNOWN ({config.prox_logp_method})"
            )
            logger.info(f"  Proximal policy (π_prox): {desc}")

            logger.info("  log_p_theta (π_θ): TRAINING FORWARD PASS (current policy)")

            if config.rejection_sampling is not None:
                rs = config.rejection_sampling
                logger.info(
                    f"  Rejection sampling: level={rs.level}, metric={rs.metric}, "
                    f"action={rs.action}, upper={rs.upper}"
                    + (f", lower={rs.lower}" if rs.lower is not None else "")
                    + (f", agg={rs.agg}" if rs.level == "sequence" else "")
                )

        # Log other critical config
        logger.info("=" * 70)
        logger.info("Training Parameters:")
        logger.info(
            f"  importance_sampling_level: {getattr(config, 'importance_sampling_level', 'token')}"
        )
        logger.info(
            f"  adv_norm: {config.adv_norm if config.adv_norm else 'DISABLED (None)'}"
        )
        logger.info(
            f"  reward_norm: {config.reward_norm if config.reward_norm else 'DISABLED (None)'}"
        )
        logger.info(f"  gae_lambda: {config.gae_