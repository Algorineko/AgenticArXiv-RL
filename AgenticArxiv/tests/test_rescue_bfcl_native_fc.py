from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from rescue_bfcl_native_fc import canonical_schema,parse_calls

def test_native_parallel_calls_and_argument_string():
    text='<tool_call>{"name":"pwd","arguments":{}}</tool_call>\n<tool_call>{"name":"ls","arguments":"{\\"a\\":true}"}</tool_call>'
    assert parse_calls(text)==[{'name':'pwd','args':{}},{'name':'ls','args':{'a':True}}]
    assert parse_calls('Done.')==[]

def test_schema_conversion_preserves_property_named_type_and_descriptions():
    source={'type':'dict','properties':{'type':{'type':'string','description':'unit'},'a':{'type':'boolean'}},'required':['type']}
    converted=canonical_schema(source)
    assert converted['type']=='object' and converted['properties']==source['properties'] and converted['required']==['type']
    assert source['type']=='dict'

@pytest.mark.parametrize('text',['<tool_call>{"name":"ls","arguments":{}}','<tool_call>{"name":"ls","args":{}}</tool_call>'])
def test_incomplete_or_wrong_protocol_rejected(text):
    with pytest.raises(ValueError):parse_calls(text)
