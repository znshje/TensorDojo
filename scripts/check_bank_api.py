"""End-to-end bank API/CLI checks, isolated from real questions and progress."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import Request,urlopen
from urllib.error import HTTPError
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='dojo-bank-e2e-') as temp:
        env={**os.environ,'DOJO_DATA_DIR':temp,'DOJO_BANK_PATH':str(Path(temp)/'question-bank.json')}
        proc=None
        def start():
            p=subprocess.Popen([sys.executable,str(ROOT/'backend/server.py'),'--port','0'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,cwd=ROOT)
            line=p.stdout.readline().strip()
            if not line.startswith('Tensor Dojo: '):raise RuntimeError(p.stderr.read())
            return p,line.split('Tensor Dojo: ')[1]
        try:
            proc,base=start();token=''
            def api(path,method='GET',body=None,auth=True):
                headers={'Content-Type':'application/json'}
                if auth:headers['X-Dojo-Token']=token
                req=Request(base+path,method=method,headers=headers,data=None if body is None else json.dumps(body).encode())
                try:
                    with urlopen(req,timeout=50) as response:return response.status,json.load(response)
                except HTTPError as exc:return exc.code,json.load(exc)
            token=api('/api/health')[1]['token']
            assert api('/api/bank',auth=False)[0]==403
            bundle=json.loads((ROOT/'examples/question-bank.json').read_text());id=bundle['problems'][0]['id']
            assert api('/api/bank')[1]['revision']==0
            assert api('/api/bank/validate','POST',{'bundle':bundle})[1]['valid']
            assert api('/api/bank/import','POST',{'bundle':bundle,'dry_run':True,'expected_revision':0})[1]['dry_run']
            assert api('/api/bank')[1]['revision']==0
            status,result=api('/api/bank/problems','POST',{'problem':bundle['problems'][0],'expected_revision':0})
            assert status==201,(status,result)
            assert api('/api/problems/'+id)[1]['totalTests']==3
            exported=api('/api/bank/export?ids='+id)[1]
            assert exported['problems'][0]['id']==id
            assert api('/api/bank/import','POST',{'bundle':bundle,'expected_revision':1})[0]==409
            changed=copy.deepcopy(bundle['problems'][0]);changed['title']='API 更新测试'
            assert api('/api/bank/problems/'+id,'PUT',{'problem':changed,'expected_revision':0})[0]==409
            assert api('/api/bank/problems/'+id,'PUT',{'problem':changed,'expected_revision':1})[0]==200
            assert api('/api/problems/'+id)[1]['title']=='API 更新测试'
            status,result=api('/api/judge','POST',{'problem_id':id,'mode':'submit','code':changed['solution']})
            assert status==200 and result['status']=='accepted' and result['total']==3,result
            # A later invalid question invalidates the whole batch.
            bad=copy.deepcopy(bundle);bad['problems'][0]['id']='other-good'
            broken=copy.deepcopy(changed);broken['id']='other-bad';broken['tests'][0]['expected']['data'][0]=999
            bad['problems'].append(broken)
            assert api('/api/bank/import','POST',{'bundle':bad,'expected_revision':2})[0]==422
            assert api('/api/bank')[1]['revision']==2
            assert api('/api/problems/other-good')[0]==404
            assert api('/api/bank/problems/'+id,'DELETE',{'expected_revision':2})[0]==200
            assert api('/api/problems/'+id)[0]==404
            assert api('/api/bank/problems/'+id+'/restore','POST',{'expected_revision':3})[0]==200
            assert api('/api/progress')[1][id]['solved']==1
            out=Path(temp)/'export.json'
            cli=subprocess.run([sys.executable,str(ROOT/'scripts/bank_cli.py'),'--url',base,'export','--ids',id,'-o',str(out)],capture_output=True,text=True)
            assert cli.returncode==0,cli.stderr
            assert json.loads(out.read_text())['problems'][0]['title']=='API 更新测试'
            proc.terminate();proc.wait(timeout=5)
            proc,base=start();token=api('/api/health')[1]['token']
            assert api('/api/bank')[1]['revision']==4
            assert api('/api/problems/'+id)[1]['title']=='API 更新测试'
            assert api('/api/judge','POST',{'problem_id':id,'mode':'run','code':changed['solution']})[1]['status']=='accepted'
            print('PASS: isolated HTTP auth, validate, dry-run, create, update, conflicts, atomic failure, export/CLI, delete/restore, judging and restart persistence.')
        finally:
            if proc and proc.poll() is None:proc.terminate();proc.wait(timeout=5)
