import sys, json, warnings
sys.path.insert(0,'.')
warnings.simplefilter("ignore")
import probe_preserving_dataset as P
ns,_=P.build_namespace()
OFD=ns["OpenAIFormatDataset"]
from pathlib import Path
import tempfile, datasets
with tempfile.TemporaryDirectory() as t:
    p=Path(t)/'h.jsonl'; P.write_jsonl(p, P.sample_hetero_tools())
    off=OFD(str(p), use_preserving_dataset=False); on=OFD(str(p), use_preserving_dataset=True)
    a=json.dumps(off.dataset[0]["tools"], sort_keys=True); b=json.dumps(on.dataset[0]["tools"], sort_keys=True)
    fn0=P.tool_functions([off.dataset[0]])[0]; fn0_on=P.tool_functions([on.dataset[0]])[0]
    print(json.dumps({"datasets":datasets.__version__,"same":a==b,"off_fn0":fn0,"on_fn0_keys":sorted(fn0_on)}, ensure_ascii=False))
