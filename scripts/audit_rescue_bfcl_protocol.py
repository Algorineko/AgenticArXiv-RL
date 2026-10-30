"""Count schema information removed by the current static BFCL port."""
import hashlib,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'AgenticArxiv'))
from benchmark.tasks_bfcl import _bfcl_signature

def audit(path):
 frame=pd.read_parquet(path)
 counts={'rows':len(frame),'tool_instances':0,'defined_parameter_instances':0,'required_parameter_instances':0,'required_parameter_descriptions':0,'parameter_descriptions':0,'parameter_enums':0,'nonempty_response_schemas':0}
 for row in frame.itertuples():
  for tool in row.tools:
   counts['tool_instances']+=1
   counts['nonempty_response_schemas']+=bool(tool.get('response'))
   params=tool.get('parameters') or {};props=params.get('properties') or {};req=params.get('required')
   req=set(list(req)) if req is not None else set()
   for name,spec in props.items():
    if not isinstance(spec,dict):continue
    counts['defined_parameter_instances']+=1
    counts['parameter_descriptions']+=bool(spec.get('description'))
    counts['parameter_enums']+=spec.get('enum') is not None
    if name in req:
     counts['required_parameter_instances']+=1
     counts['required_parameter_descriptions']+=bool(spec.get('description'))
 return {'kind':'static_bfcl_port_protocol_audit','source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'columns':list(frame.columns),'counts':counts,
         'port_has_live_backend':False,'port_action':'submit_calls entire chain, return string-match reward components',
         'port_schema_rendering':'function-level description; required names; optional names/types; no parameter descriptions/enums or response schema',
         'limits':['No historical training/evaluation traces rescored','No claim that every task is unsolvable','This is a derivative static chain-prediction task, not official BFCL multi-turn backend/state evaluation']}
if __name__=='__main__':
 result=audit(Path(sys.argv[1]));Path(sys.argv[2]).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result['counts']))
