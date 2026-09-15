"""Run submitted reference code only in a bounded, disposable process."""
import json
from pathlib import Path
import sys
import traceback
import torch
torch.set_num_threads(1)
from bank import executable,decode
from catalog import PROBLEMS,public_problem
from templates import add_templates
from worker import judge

try:
    import resource
except ImportError:
    resource = None


def prepare(spec):
    p=executable(spec)
    for test in spec['tests']:
        args=tuple(decode(v) for v in test['args'])
        expected=decode(test['expected'])
        with torch.no_grad():actual=p['reference'](*args)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-7,check_dtype=False)
    PROBLEMS[p['id']]=p
    result=judge(p['id'],p['solution'])
    if result['status']!='accepted':raise ValueError('参考实现无法通过判题：'+json.dumps(result,ensure_ascii=False)[:4000])
    add_templates({p['id']:p})
    if len(p['starter'].encode())>95000:raise ValueError('生成模板过大，请减小第一个公开测试的张量尺寸')
    public=public_problem(p,True)
    return {'spec':spec,'prepared':{k:p[k] for k in ('starter','runner','solution')}|{'public':public}}

if __name__=='__main__':
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_CPU,(25,26))
        resource.setrlimit(resource.RLIMIT_FSIZE,(20*1024*1024,20*1024*1024))
    output=Path(sys.argv[2]);results=[]
    try:
        for spec in json.loads(Path(sys.argv[1]).read_text()):
            try:results.append(prepare(spec))
            except Exception as exc:raise ValueError(spec['id']+'：'+str(exc)) from exc
        output.write_text(json.dumps({'rows':results},ensure_ascii=False,allow_nan=False))
    except Exception:
        output.write_text(json.dumps({'error':traceback.format_exc(limit=3)[-5000:]},ensure_ascii=False))
