"""Standalone runnable scaffolds derived from each problem's first public case."""
import pprint
import ast
import torch


def literal(value):
    if isinstance(value,torch.Tensor):
        return 'torch.tensor('+pprint.pformat(value.tolist(),width=100,compact=True)+', dtype='+str(value.dtype)+').reshape('+repr(tuple(value.shape))+')'
    if isinstance(value,tuple): return '('+', '.join(literal(x) for x in value)+(',' if len(value)==1 else '')+')'
    return repr(value)


def add_templates(problems):
    # Preserve the caller's RNG and avoid initializing a CUDA context.
    with torch.random.fork_rng(devices=[]):
        for p in problems.values():
            torch.manual_seed(17)
            name,args=p['cases']()[0]
            with torch.no_grad(): expected=p['reference'](*args)
            parsed=ast.parse('def solve('+p['signature']+'):\n    pass').body[0].args
            names=[arg.arg for arg in parsed.posonlyargs+parsed.args]
            assignments=[]
            for name,value in zip(names,args):
                expression=literal(value)
                assignments.append('    '+name+' = '+expression.replace('\n','\n    '))
            expected_code=literal(expected).replace('\n','\n    ')
            runner='\n\n# ===== 本地快速运行：python solution.py =====\n'
            runner+='if __name__ == "__main__":\n    torch.set_num_threads(1)\n    torch.set_printoptions(precision=5, linewidth=100)\n'
            runner+='\n'.join(assignments)+'\n'
            runner+='    expected = '+expected_code+'\n'
            runner+='    try:\n        actual = solve('+', '.join(names[:len(args)])+')\n'
            runner+='    except NotImplementedError:\n        print("模板已就绪：请先补全 solve，再运行本文件。")\n'
            runner+='    else:\n        print("实际输出:", actual)\n        print("期望输出:", expected)\n'
            runner+='        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)\n'
            runner+='        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")\n'
            p['runner']=runner
            if not has_main(p['starter']): p['starter']+=runner
            if not has_main(p['solution']): p['solution']+=runner


def has_main(code):
    return any(isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__' and any(isinstance(v,ast.Constant) and v.value=='__main__' for v in n.test.comparators) for n in ast.parse(code).body)
