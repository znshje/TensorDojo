"""Regression tests for algorithm correctness and the local judge contract."""
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
import torch
torch.set_num_threads(1)
from catalog import PROBLEMS
from worker import judge
from server import run_submission

class JudgeTests(unittest.TestCase):
    def test_all_references_and_starters(self):
        for p in PROBLEMS.values():
            with self.subTest(problem=p['id']):
                result=judge(p['id'],p['solution'])
                self.assertEqual(result['status'],'accepted',result)
                self.assertGreaterEqual(result['total'],6)
                self.assertNotEqual(judge(p['id'],p['starter'],'run')['status'],'accepted')

    def test_wrong_shape_and_detached_gradients_fail(self):
        for code in ['import torch\ndef solve(x,dim=-1): return torch.zeros(1)',
                     'import torch\ndef solve(x,dim=-1):\n z=x-x.max(dim=dim,keepdim=True).values\n return (z.exp()/z.exp().sum(dim=dim,keepdim=True)).detach()']:
            self.assertEqual(judge('softmax',code)['status'],'wrong_answer')

    def test_clipping_sign_error_fails(self):
        code='import torch\ndef solve(new_logp,old_logp,advantages,clip_eps=.2):\n r=(new_logp-old_logp.detach()).exp()\n return -(r.clamp(1-clip_eps,1+clip_eps)*advantages.detach()).mean()'
        self.assertEqual(judge('ppo',code)['status'],'wrong_answer')

    def test_group_normalization_wrong_axis_fails(self):
        code='import torch\ndef solve(rewards,eps=1e-6):\n r=rewards.detach()\n return (r-r.mean())/(r.std(unbiased=False)+eps)'
        self.assertEqual(judge('grpo-advantages',code)['status'],'wrong_answer')

    def test_errors_and_forbidden_operators(self):
        self.assertEqual(judge('softmax','def nope(')['status'],'error')
        self.assertEqual(judge('softmax','import torch\ndef solve(x,dim=-1): return torch.softmax(x,dim)')['status'],'error')
        self.assertEqual(judge('softmax','from torch import softmax as s\ndef solve(x,dim=-1): return s(x,dim)')['status'],'error')
        self.assertEqual(judge('mse','def solve(pred,target): return 0.0')['status'],'wrong_answer')

    def test_output_is_bounded(self):
        result=judge('mse','import torch\nprint("x"*20000)\ndef solve(pred,target): return ((pred-target)**2).mean()','run')
        self.assertEqual(result['status'],'accepted')
        self.assertEqual(len(result['stdout']),16000)

    def test_actual_subprocess(self):
        result=run_submission({'problem_id':'mse','code':PROBLEMS['mse']['solution'],'mode':'submit'})
        self.assertEqual(result['status'],'accepted',result)

    def test_infinite_loop_timeout(self):
        result=run_submission({'problem_id':'mse','code':'while True: pass','mode':'run'},timeout=4)
        self.assertEqual(result['status'],'timeout',result)

if __name__=='__main__': unittest.main(verbosity=2)
