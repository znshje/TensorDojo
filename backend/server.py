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
from urllib.parse import urlparse
import torch
torch.set_num_threads(1)
from catalog import PROBLEMS,public_problem

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'
TOKEN=secrets.token_urlsafe(32)
SLOTS=threading.BoundedSemaphore(2)

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
        proc=subprocess.Popen([sys.executable,str(ROOT/'backend/worker.py'),str(request),str(result)],cwd=temp,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        try:
            proc.wait(timeout=timeout)
            if proc.returncode!=0 or not result.exists():
                return {'status':'error','error':f'判题进程异常退出（exit={proc.returncode}），可能超出 CPU 限制或发生运行时错误。','cases':[]}
            try: return json.loads(result.read_text())
            except (ValueError,OSError): return {'status':'error','error':'判题结果不可读','cases':[]}
        except subprocess.TimeoutExpired:
            return {'status':'timeout','error':f'超过 {timeout} 秒运行时限，请检查死循环或过大的张量。','cases':[]}
        finally:
            try: os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait()

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
        if urlparse(self.path).path!='/api/judge': return self.json(404,{'error':'接口不存在'})
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n<1 or n>100000: return self.json(413,{'error':'代码请求大小须在 1–100000 字节之间'})
            payload=json.loads(self.rfile.read(n))
            if not isinstance(payload,dict) or payload.get('problem_id') not in PROBLEMS or not isinstance(payload.get('code'),str) or payload.get('mode') not in ('run','submit','demo'): raise ValueError()
        except (ValueError,TypeError): return self.json(400,{'error':'判题参数不合法'})
        if not SLOTS.acquire(blocking=False): return self.json(429,{'error':'判题队列已满，请稍后重试'})
        try:
            result=run_submission(payload)
            if payload['mode']=='submit':
                with database() as db:
                    cur=db.execute('INSERT INTO submissions(problem_id,code,status,result) VALUES (?,?,?,?)',(payload['problem_id'],payload['code'],result['status'],json.dumps(result)))
                    result['submission_id']=cur.lastrowid
            self.json(200,result)
        finally: SLOTS.release()

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8765); args=parser.parse_args()
    database().close()
    print(f'Tensor Dojo: http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
