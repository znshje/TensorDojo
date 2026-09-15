"""Per-submission subprocess. This is a local runner, not a hostile-code sandbox."""
import ast
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import torch
from catalog import PROBLEMS

try:
    import resource
except ImportError:
    resource = None

torch.set_num_threads(1)


class BoundedLog(io.StringIO):
    def write(self,s):
        remaining=16000-self.tell()
        if remaining>0: super().write(s[:remaining])
        return len(s)

def clone(args, constant_indices=()):
    return tuple(x.detach().clone().requires_grad_(x.is_floating_point() and i not in constant_indices) if isinstance(x,torch.Tensor) else x for i,x in enumerate(args))

def flatten(x):
    if isinstance(x,torch.Tensor): return [x]
    if isinstance(x,(tuple,list)): return sum((flatten(v) for v in x),[])
    raise AssertionError('返回值必须是 Tensor 或 Tensor 元组，不能用 item() 转成 Python 数值')

def compare(candidate,reference,args,constant_indices=(),immutable=False):
    actual_args,expected_args=clone(args,constant_indices),clone(args,constant_indices)
    before=[x.detach().clone() if isinstance(x,torch.Tensor) else x for x in actual_args]
    actual=candidate(*actual_args)
    if immutable:
        for old,new in zip(before,actual_args):
            if isinstance(old,torch.Tensor) and not torch.equal(old,new.detach()):
                raise AssertionError("不能原地修改输入张量或缓存")
    expected=reference(*expected_args)
    if isinstance(expected,tuple) and (not isinstance(actual,tuple) or len(actual)!=len(expected)):
        raise AssertionError('本题要求返回指定长度的 tuple')
    if isinstance(expected,torch.Tensor) and not isinstance(actual,torch.Tensor):
        raise AssertionError('本题要求返回单个 Tensor')
    aa,ee=flatten(actual),flatten(expected)
    if len(aa)!=len(ee): raise AssertionError('返回张量数量不正确')
    for a,e in zip(aa,ee):
        if a.shape!=e.shape: raise AssertionError(f'形状错误：期望 {list(e.shape)}，实际 {list(a.shape)}')
        torch.testing.assert_close(a,e,rtol=1e-5,atol=1e-7,check_dtype=False)
        if a.requires_grad!=e.requires_grad: raise AssertionError('梯度状态不正确：检查 detach/no_grad 或意外的梯度截断')
    active=[(a,e) for a,e in zip(aa,ee) if e.requires_grad]
    if active:
        # Nonuniform probes catch incorrect Jacobians that a simple sum can miss.
        probes=[torch.randn_like(e) for _,e in active]
        loss_a=sum((a*p).sum() for (a,e),p in zip(active,probes))
        loss_e=sum((e*p).sum() for (a,e),p in zip(active,probes))
        inputs_a=[x for x in actual_args if isinstance(x,torch.Tensor) and x.requires_grad]
        inputs_e=[x for x in expected_args if isinstance(x,torch.Tensor) and x.requires_grad]
        ga=torch.autograd.grad(loss_a,inputs_a,allow_unused=True)
        ge=torch.autograd.grad(loss_e,inputs_e,allow_unused=True)
        for a,e in zip(ga,ge):
            if e is None:
                if a is not None and torch.count_nonzero(a): raise AssertionError('固定目标/旧策略不应收到梯度')
            elif a is None: raise AssertionError('可训练输入的梯度被截断')
            else: torch.testing.assert_close(a,e,rtol=2e-5,atol=1e-7,check_dtype=False)
    return [list(x.shape) for x in aa]

def judge(problem_id,code,mode='submit'):
    problem=PROBLEMS[problem_id]
    log=BoundedLog(); results=[]
    started=time.perf_counter()
    try:
        tree=ast.parse(code,filename='solution.py')
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                if name in problem['forbidden']: raise ValueError(f'练习限制：请手写 {name}，不要直接调用该函数')
            if isinstance(node,ast.ImportFrom) and any(n.name in problem['forbidden'] for n in node.names):
                raise ValueError('练习限制：不能导入本题禁止的现成算子')
        namespace={'__name__':'__main__' if mode=='demo' else '__solution__'}
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            exec(compile(tree,'solution.py','exec'),namespace)
        if mode=='demo':
            return {'status':'executed','cases':[],'stdout':log.getvalue(),'ms':round((time.perf_counter()-started)*1000,1)}
        fn=namespace.get('solve')
        if not callable(fn): raise ValueError('请定义可调用的 solve 函数')
        # Public run uses one seed; submit checks public + two independent random seeds.
        for seed in ([17] if mode=='run' else problem.get('seeds',[17,271,901])):
            torch.manual_seed(seed)
            cases=problem['cases']()
            if mode=='run': cases=cases[:2]
            for i,(name,args) in enumerate(cases):
                start=time.perf_counter()
                try:
                    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                        constant_indices=tuple(range(len(args))) if problem.get('inference_only') else tuple(problem.get('constant_indices',((1,) if problem_id in ('mse','bce','focal') else ())))
                        shapes=compare(fn,problem['reference'],args,constant_indices,problem.get('immutable_inputs',False))
                    results.append({'name':name if seed==17 else f'随机回归 {seed} · {i+1}','passed':True,'shapes':shapes,'ms':round((time.perf_counter()-start)*1000,2)})
                except Exception as exc:
                    results.append({'name':name if seed==17 else f'随机回归 {seed} · {i+1}','passed':False,'error':str(exc)[:2500],'ms':round((time.perf_counter()-start)*1000,2)})
    except Exception:
        return {'status':'error','error':traceback.format_exc(limit=5)[-4000:],'cases':results,'stdout':log.getvalue()}
    passed=sum(r['passed'] for r in results)
    return {'status':'accepted' if passed==len(results) and results else 'wrong_answer','passed':passed,'total':len(results),'cases':results,'stdout':log.getvalue(),'ms':round((time.perf_counter()-started)*1000,1)}

if __name__=='__main__':
    if resource is not None and os.name=='posix':
        resource.setrlimit(resource.RLIMIT_CPU,(10,11))
        resource.setrlimit(resource.RLIMIT_FSIZE,(2*1024*1024,2*1024*1024))
    request=json.loads(Path(sys.argv[1]).read_text())
    if '_bank_spec' in request:
        from bank import executable
        PROBLEMS[request['problem_id']]=executable(request['_bank_spec'])
    result=judge(request['problem_id'],request['code'],request.get('mode','submit'))
    Path(sys.argv[2]).write_text(json.dumps(result,ensure_ascii=False))
