"""Versioned JSON question bank. Python execution stays in validation/judge children."""
import ast
import copy
import json
import math
import os
from pathlib import Path
import re
import threading
import torch

FORMAT='tensor-dojo-bank'
VERSION=1
BANK_PATH=Path(os.environ.get('DOJO_BANK_PATH',Path(os.environ.get('DOJO_DATA_DIR',Path(__file__).resolve().parent.parent/'data'))/'question-bank.json'))
DTYPES={k:getattr(torch,k) for k in ('float32','float64','int32','int64','bool')}
FIELDS=('id','title','category','difficulty','minutes','description','signature','formula','hints','source','starter','solution','forbidden','inference_only','immutable_inputs','constant_indices','tests')

class BankError(ValueError):
    def __init__(self,message,status=400):super().__init__(message);self.status=status


def encode(value):
    if isinstance(value,torch.Tensor):
        return {'type':'tensor','dtype':str(value.dtype).split('.')[-1],'shape':list(value.shape),'data':value.detach().cpu().reshape(-1).tolist()}
    if isinstance(value,tuple):return {'type':'tuple','items':[encode(v) for v in value]}
    if value is None or isinstance(value,(str,bool,int,float)):return value
    raise BankError('仅支持 Tensor、tuple、标量与 None')


def decode(value):
    if isinstance(value,dict):
        if value.get('type')=='tuple':
            items=value.get('items')
            if not isinstance(items,list) or len(items)>32:raise BankError('tuple.items 必须为最多 32 项的列表')
            return tuple(decode(v) for v in items)
        if value.get('type')!='tensor' or value.get('dtype') not in DTYPES:raise BankError('不支持的张量类型')
        shape=value.get('shape');data=value.get('data')
        if not isinstance(shape,list) or len(shape)>8 or any(type(n)!=int or n<0 or n>100000 for n in shape):raise BankError('非法 tensor.shape')
        if not isinstance(data,list) or len(data)!=math.prod(shape) or len(data)>100000:raise BankError('tensor.data 必须为与 shape 相符的扁平数组，最多 100000 元素')
        if any(type(v) not in (int,float,bool) or not math.isfinite(v) for v in data):raise BankError('张量不能包含 NaN、Infinity 或非数值')
        if value['dtype']=='bool' and any(type(v)!=bool for v in data):raise BankError('bool 张量数据必须为 true/false')
        if value['dtype'] in ('int32','int64') and any(type(v)!=int for v in data):raise BankError('整数张量只能包含整数')
        try:return torch.tensor(data,dtype=DTYPES[value['dtype']]).reshape(shape)
        except (ValueError,TypeError,RuntimeError,OverflowError) as exc:raise BankError('张量不可构造：'+str(exc)) from exc
    if value is None or type(value) in (str,bool,int):return value
    if type(value)==float and math.isfinite(value):return value
    raise BankError('参数需为合法标量、tensor 或 tuple 编码')


def validate_spec(raw):
    if not isinstance(raw,dict):raise BankError('每道题必须为 JSON 对象')
    extra=set(raw)-set(FIELDS)
    if extra:raise BankError('未知题目字段：'+', '.join(sorted(extra)))
    p=copy.deepcopy(raw)
    for key in ('id','title','category','difficulty','description','signature','solution'):
        if not isinstance(p.get(key),str) or not p[key].strip():raise BankError(f'{key} 必须为非空字符串')
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,63}',p['id']):raise BankError('id 只能是 1–64 位小写字母、数字、连字符，首位为字母')
    if p['difficulty'] not in ('入门','基础','进阶','挑战'):raise BankError('difficulty 必须为 入门/基础/进阶/挑战')
    if type(p.get('minutes'))!=int or not 1<=p['minutes']<=600:raise BankError('minutes 必须为 1–600 的整数')
    for key in ('formula','source'):p.setdefault(key,'')
    for key in ('formula','source','title','category','description','signature'):
        if not isinstance(p[key],str) or len(p[key])>20000:raise BankError(f'{key} 长度或类型不合法')
    if p['source'] and not p['source'].startswith(('https://','http://')):raise BankError('source 只允许 http(s) 链接')
    for key in ('hints','forbidden'):
        p.setdefault(key,[])
        if not isinstance(p[key],list) or len(p[key])>30 or any(not isinstance(x,str) or len(x)>4000 for x in p[key]):raise BankError(f'{key} 必须为字符串数组')
    for key in ('inference_only','immutable_inputs'):
        p.setdefault(key,False)
        if type(p[key])!=bool:raise BankError(f'{key} 必须为 bool')
    p.setdefault('constant_indices',[])
    if not isinstance(p['constant_indices'],list) or any(type(v)!=int or not 0<=v<32 for v in p['constant_indices']):raise BankError('constant_indices 必须为非负参数下标数组')
    try:
        signature=ast.parse('def solve('+p['signature']+'):\n    pass').body[0]
        if not isinstance(signature,ast.FunctionDef) or signature.args.vararg or signature.args.kwarg or signature.args.kwonlyargs:raise BankError('signature 只支持固定的位置参数与默认值')
        defaults=signature.args.defaults
        for d in defaults:ast.literal_eval(d)
        argc=len(signature.args.posonlyargs)+len(signature.args.args)
        if not 1<=argc<=32:raise BankError('参数数目须在 1–32 之间')
        if any(i>=argc for i in p['constant_indices']):raise BankError('constant_indices 超出参数范围')
        p.setdefault('starter','import torch\nimport math\n\ndef solve('+p['signature']+'):\n    raise NotImplementedError("请补全 solve")\n')
        for key in ('starter','solution'):
            if not isinstance(p[key],str) or len(p[key].encode())>80000:raise BankError(f'{key} 代码最多 80000 字节')
            tree=ast.parse(p[key])
            if not any(isinstance(n,ast.FunctionDef) and n.name=='solve' for n in tree.body):raise BankError(f'{key} 需定义顶层 solve 函数')
    except (SyntaxError,ValueError,TypeError) as exc:raise BankError('代码或签名无效：'+str(exc)) from exc
    tests=p.get('tests')
    if not isinstance(tests,list) or not 1<=len(tests)<=200:raise BankError('tests 需要 1–200 个测试用例')
    for t in tests:
        if not isinstance(t,dict) or set(t)!={'name','args','expected'}:raise BankError('每个测试需包含且仅包含 name、args、expected')
        if not isinstance(t['name'],str) or not t['name'] or len(t['name'])>200:raise BankError('测试名称无效')
        if not isinstance(t['args'],list) or not argc-len(defaults)<=len(t['args'])<=argc:raise BankError('测试参数数目与 signature 不符')
        for arg in t['args']:decode(arg)
        expected=decode(t['expected'])
        def valid_output(x):return isinstance(x,torch.Tensor) or isinstance(x,tuple) and len(x)>0 and all(valid_output(v) for v in x)
        if not valid_output(expected):raise BankError('expected 必须为 Tensor 或非空 Tensor 元组')
    return p


def validate_bundle(bundle):
    if not isinstance(bundle,dict) or bundle.get('format')!=FORMAT or type(bundle.get('version'))!=int or bundle['version']!=VERSION:raise BankError('需要 format=tensor-dojo-bank、version=1')
    rows=bundle.get('problems')
    if not isinstance(rows,list) or not 1<=len(rows)<=200:raise BankError('problems 需要 1–200 道题')
    validated=[];seen=set()
    for raw in rows:
        p=validate_spec(raw)
        if p['id'] in seen:raise BankError('导入文件包含重复 id：'+p['id'])
        seen.add(p['id']);validated.append(p)
    return validated


def executable(spec,prepared=None):
    """Construct lazy reference; do not execute user Python in the HTTP process."""
    cache={}
    def reference(*args):
        if 'solve' not in cache:
            ns={'__name__':'__reference__'}
            exec(compile(spec['solution'],spec['id']+'.py','exec'),ns)
            cache['solve']=ns['solve']
        return cache['solve'](*args)
    def cases():return [(t['name'],tuple(decode(v) for v in t['args'])) for t in spec['tests']]
    p={**spec,'reference':reference,'cases':cases,'seeds':[17]}
    if prepared:
        p.update({k:prepared[k] for k in ('starter','runner','solution')})
        p['_public']=prepared['public']
    return p


def export_problem(p):
    if '_spec' in p:return copy.deepcopy(p['_spec'])
    spec={k:copy.deepcopy(p[k]) for k in FIELDS if k in p and k!='tests'}
    spec['forbidden']=list(spec.get('forbidden',[]))
    spec['constant_indices']=[1] if p['id'] in ('mse','bce','focal') else []
    spec['tests']=[]
    with torch.random.fork_rng(devices=[]):
        for seed in p.get('seeds',[17,271,901]):
            torch.manual_seed(seed)
            for name,args in p['cases']():
                with torch.no_grad():expected=p['reference'](*args)
                spec['tests'].append({'name':name if seed==17 else f'随机回归 {seed} · {name}','args':[encode(v) for v in args],'expected':encode(expected)})
    return spec


def read_state(path=BANK_PATH):
    if not path.exists():return {'revision':0,'records':{}}
    state=json.loads(path.read_text())
    if not isinstance(state.get('records'),dict) or type(state.get('revision'))!=int:raise BankError('题库存储文件损坏',500)
    return state


def apply_overlay(problems,path=BANK_PATH):
    for id,record in read_state(path)['records'].items():
        if record['deleted']:problems.pop(id,None)
        else:
            p=executable(record['spec'],record['prepared']);p['_spec']=record['spec'];problems[id]=p


class BankStore:
    def __init__(self,builtins,path=BANK_PATH):
        self.path=path;self.builtins=builtins;self.lock=threading.RLock()
        self.state=read_state(path)
        self._problems=dict(builtins);apply_overlay(self._problems,path)
    def snapshot(self):
        with self.lock:return dict(self._problems)
    def info(self):
        with self.lock:
            rows=[{'id':id,'title':p['title'],'category':p['category'],'customized':id in self.state['records']} for id,p in self._problems.items()]
            deleted=[id for id,r in self.state['records'].items() if r['deleted']]
            return {'revision':self.state['revision'],'problems':rows,'deleted':deleted}
    def get(self,id):
        with self.lock:
            if id not in self._problems:raise BankError('题目不存在',404)
            return {'revision':self.state['revision'],'problem':export_problem(self._problems[id])}
    def export(self,ids=None):
        with self.lock:
            chosen=list(self._problems) if ids is None else ids
            if not chosen or any(id not in self._problems for id in chosen):raise BankError('导出题目不存在或为空',404)
            return {'format':FORMAT,'version':VERSION,'problems':[export_problem(self._problems[id]) for id in dict.fromkeys(chosen)]}
    def check_revision(self,revision):
        if type(revision)!=int or revision!=self.state['revision']:raise BankError('题库版本已变化；请重新读取 revision 后重试',409)
    def _save(self,state,problems):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        import tempfile
        fd,tmp=tempfile.mkstemp(dir=self.path.parent,prefix='.bank-',suffix='.json')
        try:
            with os.fdopen(fd,'w') as out:
                json.dump(state,out,ensure_ascii=False,allow_nan=False);out.flush();os.fsync(out.fileno())
            os.replace(tmp,self.path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        self.state=state;self._problems=problems
    def import_prepared(self,rows,revision,overwrite=False,dry_run=False):
        with self.lock:
            self.check_revision(revision)
            conflicts=[r['spec']['id'] for r in rows if r['spec']['id'] in self._problems or r['spec']['id'] in self.state['records']]
            if conflicts and not overwrite:raise BankError('存在同名题目；需要 overwrite=true：'+', '.join(conflicts),409)
            result={'validated':len(rows),'ids':[r['spec']['id'] for r in rows],'overwritten':conflicts,'dry_run':dry_run,'revision':revision}
            if dry_run:return result
            state=copy.deepcopy(self.state);problems=dict(self._problems)
            for row in rows:
                spec=row['spec'];id=spec['id']
                state['records'][id]={'spec':spec,'prepared':row['prepared'],'deleted':False}
                p=executable(spec,row['prepared']);p['_spec']=spec;problems[id]=p
            state['revision']+=1;self._save(state,problems);result['revision']=state['revision'];return result
    def delete(self,id,revision):
        with self.lock:
            self.check_revision(revision)
            if id not in self._problems:raise BankError('题目不存在',404)
            state=copy.deepcopy(self.state)
            if id not in state['records']:state['records'][id]={'deleted':True,'builtin':True}
            else:state['records'][id]['deleted']=True
            problems=dict(self._problems);del problems[id];state['revision']+=1;self._save(state,problems)
            return {'deleted':id,'revision':state['revision'],'recoverable':True}
    def restore(self,id,revision):
        with self.lock:
            self.check_revision(revision)
            record=self.state['records'].get(id)
            if not record or not record['deleted']:raise BankError('没有可恢复的已删除题目',404)
            state=copy.deepcopy(self.state);problems=dict(self._problems)
            if record.get('builtin'):
                problems[id]=self.builtins[id];del state['records'][id]
            else:
                state['records'][id]['deleted']=False
                p=executable(record['spec'],record['prepared']);p['_spec']=record['spec'];problems[id]=p
            state['revision']+=1;self._save(state,problems)
            return {'restored':id,'revision':state['revision']}
