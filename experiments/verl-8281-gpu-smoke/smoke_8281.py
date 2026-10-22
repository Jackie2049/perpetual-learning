"""verl #8281 GPU e2e 训练冒烟（2026-10-22，4090-1，M5）。
目的：验证 fix-8281-truncation-policy（PR #8313）改动下，agent_loop 正常训练路径
不因改动报错/退化；默认 error 策略、prompt 不超长时不应触发截断/raise。
小模型 + 单 GPU + 极少量 step，不启动长训练。"""
import json, os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import ray
ray.init(ignore_reinit_error=True)

import verl
from hydra import compose, initialize_config_dir

verl_config_dir = os.path.join(os.path.dirname(verl.__file__), "trainer/config")

model_path = "/home/zxw/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
train_file = "/tmp/gsm8k_agent/train.parquet"
test_file = "/tmp/gsm8k_agent/test.parquet"

# 最小 tool_config：sandbox 远程服务不需要真实执行环境也能走通 agent loop 结构
tool_config_path = "/tmp/tool_config_8281.json"
with open(tool_config_path, "w") as f:
    json.dump({"tools": []}, f)

with initialize_config_dir(config_dir=verl_config_dir):
    config = compose(
        config_name="ppo_trainer",
        overrides=[
            "algorithm.adv_estimator=grpo",
            "data.train_files=" + train_file,
            "data.val_files=" + test_file,
            "data.return_raw_chat=True",
            "data.train_batch_size=2",
            "data.max_prompt_length=512",
            "data.max_response_length=256",
            "+data.apply_chat_template_kwargs.enable_thinking=False",
            "actor_rollout_ref.model.path=" + model_path,
            "actor_rollout_ref.actor.ppo_mini_batch_size=2",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
            "actor_rollout_ref.actor.fsdp_config.param_offload=True",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.mode=async",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.rollout.n=2",
            "actor_rollout_ref.rollout.multi_turn.tool_config_path=" + tool_config_path,
            "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
            "trainer.val_before_train=False",
            "trainer.n_gpus_per_node=1",
            "trainer.test_freq=-1",
            "trainer.total_training_steps=2",
            "trainer.logger=[console]",
            "trainer.project_name=verl",
            "trainer.experiment_name=8281_smoke",
            "trainer.default_local_dir=/tmp/8281_smoke_out",
            "trainer.save_freq=-1",
        ],
    )

from verl.trainer.main_ppo import main
main(config)
print("SMOKE COMPLETED OK")
