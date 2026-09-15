"""Loopback-only standard-library HTTP API and static server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import secrets
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
import torch
torch.set_num_threads(1)
from catalog import PROBLEMS,BUILTIN_PROBLEMS,public_problem
from bank import BankStore,BankError,validate_bundle,export_problem,FORMAT,VERSION

ROOT=Path(__file__).resolve().parent.parent
DATA=Path(os.environ.get('DOJO_DATA_DIR',ROOT/'data'))
TOKEN=secrets.token_urlsafe(32)
SLOTS=threading.BoundedSemaphore(2)
STORE=BankStore(BUILTIN_PROBLEMS)
BANK_SLOTS=threading.BoundedSemaphore(1)

def popen_options():
    options={'stdout':subprocess.DEVNULL,'stderr':subprocess.DEVNULL}
    if os.name=='posix':
        options['start_new_session']=True
    return options


def kill_process(proc):
    try:
        if os.name=='posix':
            os.killpg(proc.pid,signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError,OSError):
        pass
    try:
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired,OSError):
        pass


def database():
    DATA.mkdir(exist_ok=True)
    db=sqlite3.connect(DATA/'progress.sqlite3')
    db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE IF NOT EXISTS submissions (id INTEGER PRIMARY KEY, problem_id TEXT, code TEXT, status TEXT, result TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)')
    return db

def run_submission(payload,timeout=18):
    with tempfile.TemporaryDirectory(prefix='tensor-dojo-') as temp:
        request=Path(temp)/'request.json'; result=Path(temp)/'result.json'
        request.write_text(json.dumps(payload))
        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
        proc=subprocess.Popen([sys.executable,str(ROOT/'backend/worker.py'),str(request),str(result)],cwd=temp,env=env,**popen_options())
        try:
            proc.wait(timeout=timeout)
            if proc.returncode!=0 or not result.exists():
                return {'status':'error','error':f'判题进程异常退出（exit={proc.returncode}），可能超出 CPU 限制或发生运行时错误。','cases':[]}
            try: return json.loads(result.read_text())
            except (ValueError,OSError): return {'status':'error','error':'判题结果不可读','cases':[]}
        except subprocess.TimeoutExpired:
            return {'status':'timeout','error':f'超过 {timeout} 秒运行时限，请检查死循环或过大的张量。','cases':[]}
        finally:
            kill_process(proc)

def prepare_import(specs):
    if not BANK_SLOTS.acquire(blocking=False):raise BankError('已有题库校验正在进行，请稍后重试',429)
    try:
        with tempfile.TemporaryDirectory(prefix='dojo-bank-') as temp:
            source=Path(temp)/'request.json';result=Path(temp)/'result.json'
            source.write_text(json.dumps(specs,ensure_ascii=False))
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            proc=subprocess.Popen([sys.executable,str(ROOT/'backend/bank_validation.py'),str(source),str(result)],cwd=temp,env=env,**popen_options())
            try:
                proc.wait(timeout=40)
                if proc.returncode!=0 or not result.exists():raise BankError('参考实现校验进程异常退出或超出资源限制',422)
                response=json.loads(result.read_text())
                if 'error' in response:raise BankError(response['error'],422)
                return response['rows']
            except subprocess.TimeoutExpired:raise BankError('参考实现校验超过 40 秒，请检查代码或缩小导入批次',422)
            finally:
                kill_process(proc)
    finally:BANK_SLOTS.release()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(ROOT/'dist'),**kwargs)
    def log_message(self,fmt,*args): pass
    def json(self,status,data):
        raw=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(raw))); self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff'); self.end_headers(); self.wfile.write(raw)
    def allowed_host(self):
        host=self.headers.get('Host','')
        return host in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}','127.0.0.1:5173','localhost:5173'}
    def do_GET(self):
        if not self.allowed_host(): return self.json(403,{'error':'仅允许本机访问'})
        path=urlparse(self.path).path
        if path.startswith('/api/bank'):
            return self.bank_get(path)
        PROBLEMS=STORE.snapshot()
        if path=='/api/health': return self.json(200,{'python':sys.executable,'torch':torch.__version__,'device':'CPU','token':TOKEN})
        if path=='/api/problems': return self.json(200,[public_problem(p) for p in PROBLEMS.values()])
        if path=='/api/progress':
            with database() as db:
                rows=db.execute('SELECT problem_id, COUNT(*) attempts, MAX(status="accepted") solved FROM submissions GROUP BY problem_id').fetchall()
            return self.json(200,{r['problem_id']:dict(r) for r in rows})
        if path.startswith('/api/problems/'):
            parts=path.split('/'); id=parts[3]
            if id not in PROBLEMS: return self.json(404,{'error':'题目不存在'})
            p=PROBLEMS[id]
            if len(parts)==5 and parts[4]=='solution': return self.json(200,{'code':p['solution']})
            if len(parts)==5 and parts[4]=='history':
                with database() as db: rows=db.execute('SELECT id,status,created_at,result,code FROM submissions WHERE problem_id=? ORDER BY id DESC LIMIT 30',(id,)).fetchall()
                return self.json(200,[{**dict(r),'result':json.loads(r['result'])} for r in rows])
            return self.json(200,public_problem(p,True))
        if path.startswith('/api/'): return self.json(404,{'error':'接口不存在'})
        if path=='/' and not (ROOT/'dist/index.html').exists(): return self.json(503,{'error':'请先运行 npm run build，或使用前端开发服务器 5173'})
        return super().do_GET()
    def do_POST(self):
        if not self.allowed_host(): return self.json(403,{'error':'仅允许本机访问'})
        origin=self.headers.get('Origin')
        allowed={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}','http://127.0.0.1:5173','http://localhost:5173'}
        if origin and origin not in allowed: return self.json(403,{'error':'拒绝跨站执行请求'})
        if self.headers.get('X-Dojo-Token')!=TOKEN: return self.json(403,{'error':'会话失效，请刷新页面'})
        path=urlparse(self.path).path
        if path.startswith('/api/bank'):
            return self.bank_write(path)
        if path=='/api/complete':
            return self.complete()
        if self.command!='POST':return self.json(405,{'error':'判题只支持 POST'})
        PROBLEMS=STORE.snapshot()
        if path!='/api/judge': return self.json(404,{'error':'接口不存在'})
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n<1 or n>100000: return self.json(413,{'error':'代码请求大小须在 1–100000 字节之间'})
            payload=json.loads(self.rfile.read(n))
            if not isinstance(payload,dict) or payload.get('problem_id') not in PROBLEMS or not isinstance(payload.get('code'),str) or payload.get('mode') not in ('run','submit','demo'): raise ValueError()
        except (ValueError,TypeError): return self.json(400,{'error':'判题参数不合法'})
        if not SLOTS.acquire(blocking=False): return self.json(429,{'error':'判题队列已满，请稍后重试'})
        try:
            # Capture the chosen revision for this in-flight submission.
            payload.pop('_bank_spec',None)
            payload['_bank_spec']=export_problem(PROBLEMS[payload['problem_id']])
            result=run_submission(payload)
            if payload['mode']=='submit':
                with database() as db:
                    cur=db.execute('INSERT INTO submissions(problem_id,code,status,result) VALUES (?,?,?,?)',(payload['problem_id'],payload['code'],result['status'],json.dumps(result)))
                    result['submission_id']=cur.lastrowid
            self.json(200,result)
        finally: SLOTS.release()

    def do_PUT(self):return self.do_POST()
    def do_DELETE(self):return self.do_POST()

    def complete(self):
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 1<=n<=200000:return self.json(413,{'error':'补全请求大小不合法'})
            body=json.loads(self.rfile.read(n))
        except (ValueError,UnicodeError,TypeError):return self.json(400,{'error':'请求必须为合法 JSON'})
        if not isinstance(body,dict):return self.json(400,{'error':'请求体必须为对象'})
        code=body.get('code');line=body.get('line');column=body.get('column')
        if not isinstance(code,str) or not isinstance(line,int) or not isinstance(column,int) or not 1<=line<=100000 or not 1<=column<=100000:
            return self.json(400,{'error':'code/line/column 参数不合法'})
        try:
            import jedi
            cache=DATA/'jedi-cache';cache.mkdir(parents=True,exist_ok=True)
            os.environ.setdefault('XDG_CACHE_HOME',str(cache))
            script=jedi.Script(code=code,path='solution.py')
            items=[]
            for c in script.complete(line,column-1)[:60]:
                try:doc=(c.docstring() or '')[:500]
                except Exception:doc=''
                items.append({
                    'label':c.name,
                    'type':c.type,
                    'detail':c.type+((' · '+c.full_name) if getattr(c,'full_name',None) else ''),
                    'insertText':c.name,
                    'documentation':doc
                })
            return self.json(200,{'available':True,'items':items})
        except Exception as exc:
            return self.json(200,{'available':False,'items':[],'error':str(exc)[:200]})

    def bank_get(self,path):
        if self.headers.get('X-Dojo-Token')!=TOKEN:return self.json(403,{'error':'需要 X-Dojo-Token'})
        try:
            if path=='/api/bank':return self.json(200,STORE.info())
            if path=='/api/bank/export':
                ids=parse_qs(urlparse(self.path).query).get('ids',[None])[0]
                return self.json(200,STORE.export(ids.split(',') if ids is not None else None))
            if path=='/api/bank/example':return self.json(200,json.loads((ROOT/'examples/question-bank.json').read_text()))
            if path.startswith('/api/bank/problems/') and len(path.split('/'))==5:
                return self.json(200,STORE.get(path.split('/')[-1]))
            raise BankError('管理接口不存在',404)
        except BankError as exc:return self.json(exc.status,{'error':str(exc)})

    def bank_write(self,path):
        try:
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 1<=n<=8*1024*1024:raise BankError('题库请求必须为 1 字节至 8 MiB',413)
                body=json.loads(self.rfile.read(n))
            except (ValueError,UnicodeError) as exc:
                if isinstance(exc,BankError):raise
                raise BankError('请求必须为合法 JSON') from exc
            if not isinstance(body,dict):raise BankError('请求体必须为对象')
            revision=body.get('expected_revision')
            parts=path.split('/')
            if self.command=='DELETE' and len(parts)==5 and parts[:4]==['','api','bank','problems']:
                return self.json(200,STORE.delete(parts[4],revision))
            if self.command=='POST' and len(parts)==6 and parts[:4]==['','api','bank','problems'] and parts[5]=='restore':
                return self.json(200,STORE.restore(parts[4],revision))
            overwrite=False;dry_run=False
            if self.command=='POST' and path in ('/api/bank/import','/api/bank/validate'):
                specs=validate_bundle(body.get('bundle'))
                overwrite=body.get('overwrite',False);dry_run=body.get('dry_run',False)
                if type(overwrite)!=bool or type(dry_run)!=bool:raise BankError('overwrite/dry_run 必须为 bool')
                if path=='/api/bank/validate':
                    rows=prepare_import(specs)
                    return self.json(200,{'valid':True,'validated':len(rows),'ids':[r['spec']['id'] for r in rows]})
            elif self.command=='POST' and path=='/api/bank/problems':
                specs=validate_bundle({'format':FORMAT,'version':VERSION,'problems':[body.get('problem')]})
            elif self.command=='PUT' and len(parts)==5 and parts[:4]==['','api','bank','problems']:
                specs=validate_bundle({'format':FORMAT,'version':VERSION,'problems':[body.get('problem')]})
                if specs[0]['id']!=parts[4]:raise BankError('URL id 与题目 id 不一致')
                STORE.get(parts[4]);overwrite=True
            else:raise BankError('管理接口或 HTTP 方法不支持',404)
            with STORE.lock:STORE.check_revision(revision)
            rows=prepare_import(specs)
            result=STORE.import_prepared(rows,revision,overwrite,dry_run)
            return self.json(200 if dry_run or overwrite else 201,result)
        except BankError as exc:return self.json(exc.status,{'error':str(exc)})
        except (OSError,ValueError) as exc:return self.json(500,{'error':'题库保存/校验失败：'+str(exc)})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8765); args=parser.parse_args()
    database().close()
    httpd=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Tensor Dojo: http://127.0.0.1:{httpd.server_port}',flush=True)
    try:httpd.serve_forever()
    except KeyboardInterrupt:pass
    finally:httpd.server_close()
