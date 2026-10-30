from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from rescue_bfcl_native_preflight import parse_literal_call,execute

class Virtual:
    def sort(self,file_name):return {'virtual_file':file_name}

def test_native_dispatch_supports_literal_positional_arguments():
    assert execute(Virtual(),"sort('report.txt')",{'sort'})=={'virtual_file':'report.txt'}

@pytest.mark.parametrize('text',["__import__('os')", "sort(__import__('os').getcwd())", "sort(**{'file_name':'x'})", "sort(file_name='x',file_name='y')", "obj.sort('x')"])
def test_dispatch_rejects_executable_or_nonwhitelisted_expressions(text):
    with pytest.raises((ValueError,SyntaxError)):
        parse_literal_call(text,{'sort'})
