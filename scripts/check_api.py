"""Read-only / sample-only smoke checks against the running localhost API."""
import json
from urllib.error import HTTPError
from urllib.request import Request,urlopen
BASE='http://127.0.0.1:8765'
def request(path,data=None,headers=None):
    raw=None if data is None else json.dumps(data).encode()
    req=Request(BASE+path,data=raw,headers={'Content-Type':'application/json',**(headers or {})})
    try:
        with urlopen(req,timeout=25) as response:
            return response.status,json.load(response)
    except HTTPError as exc: return exc.code,json.load(exc)

if __name__=='__main__':
    status,health=request('/api/health'); assert status==200
    status,problems=request('/api/problems'); assert status==200 and len(problems)>0
    for p in problems:
        status,detail=request('/api/problems/'+p['id'])
        assert status==200 and '__main__' in detail['starter'] and detail['runner'] and all(e['output'] for e in detail['examples'])
    payload={'problem_id':'mse','mode':'run','code':'import torch\ndef solve(pred,target): return ((pred-target)**2).mean()'}
    assert request('/api/judge',payload)[0]==403
    headers={'X-Dojo-Token':health['token'],'Origin':'https://example.invalid'}
    assert request('/api/judge',payload,headers)[0]==403
    headers={'X-Dojo-Token':health['token'],'Origin':BASE}
    status,result=request('/api/judge',payload,headers)
    assert status==200 and result['status']=='accepted',result
    status,result=request('/api/judge',{**payload,'mode':'demo','code':'if __name__ == "__main__": print("DEMO_OK")'},headers)
    assert status==200 and result['status']=='executed' and 'DEMO_OK' in result['stdout']
    assert request('/api/judge',{**payload,'mode':'invalid'},headers)[0]==400
    assert request('/api/health',headers={'Host':'untrusted.invalid'})[0]==403
    print(f'PASS: {len(problems)} problem APIs, real Python run, token/origin/host checks, invalid request rejection.')
