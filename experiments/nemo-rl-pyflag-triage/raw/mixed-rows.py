import sys, json, warnings
sys.path.insert(0,'.')
warnings.simplefilter("ignore")
import probe_preserving_dataset as P
ns,_=P.build_namespace()
OFD=ns["OpenAIFormatDataset"]
from pathlib import Path
import tempfile, datasets
rows = P.sample_hetero_tools() + [{
  "messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}]
with tempfile.TemporaryDirectory() as t:
    p=Path(t)/'m.jsonl'; P.write_jsonl(p, rows)
    off=OFD(str(p), use_preserving_dataset=False); on=OFD(str(p), use_preserving_dataset=True)
    out=[]
    for i in range(3):
        a=json.dumps(off.dataset[i], sort_keys=True, default=str); b=json.dumps(on.dataset[i], sort_keys=True, default=str)
        out.append({"row":i,"same":a==b,"off_tools":str(off.dataset[i].get('tools','<absent>'))[:80],"on_has_tools_key":('tools' in on.dataset[i])})
    print(json.dumps({"datasets":datasets.__version__,"rows":out}, ensure_ascii=False, indent=1))
