import contextlib
import io
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
import torch
torch.set_num_threads(1)
from catalog import PROBLEMS
from worker import judge

class ExpansionTests(unittest.TestCase):
    def test_every_template_runs_as_main(self):
        for p in PROBLEMS.values():
            with self.subTest(problem=p['id']):
                log=io.StringIO()
                with contextlib.redirect_stdout(log):
                    exec(compile(p['starter'],p['id']+'.py','exec'),{'__name__':'__main__'})
                self.assertIn('请先补全 solve',log.getvalue())
                log=io.StringIO()
                with contextlib.redirect_stdout(log):
                    exec(compile(p['solution'],p['id']+'.py','exec'),{'__name__':'__main__'})
                self.assertIn('快速样例通过',log.getvalue())
                self.assertLess(len(p['starter'].encode()),100000)

    def test_demo_mode_never_awards_acceptance(self):
        for id in ('roc-auc','batchnorm2d','kv-cache-decode'):
            r=judge(id,PROBLEMS[id]['starter'],'demo')
            self.assertEqual(r['status'],'executed',r)
            self.assertIn('请先补全 solve',r['stdout'])
            r=judge(id,PROBLEMS[id]['solution'],'demo')
            self.assertEqual(r['status'],'executed',r)
            self.assertIn('快速样例通过',r['stdout'])

    def test_metrics_known_values(self):
        scores=torch.tensor([.1,.4,.35,.8],dtype=torch.float64)
        targets=torch.tensor([0,0,1,1])
        self.assertAlmostEqual(PROBLEMS['roc-auc']['reference'](scores,targets).item(),.75)
        self.assertAlmostEqual(PROBLEMS['average-precision']['reference'](scores,targets).item(),5/6)
        scores=torch.ones(4,dtype=torch.float64)
        self.assertEqual(PROBLEMS['roc-auc']['reference'](scores,targets).item(),.5)
        self.assertEqual(PROBLEMS['average-precision']['reference'](scores,targets).item(),.5)
        # Includes a false positive and false negative; recall != precision.
        s=torch.tensor([.9,.8,.2,.1]); y=torch.tensor([1,0,1,1])
        self.assertAlmostEqual(PROBLEMS['precision']['reference'](s,y).item(),.5)
        self.assertAlmostEqual(PROBLEMS['recall']['reference'](s,y).item(),1/3)
        self.assertAlmostEqual(PROBLEMS['f1']['reference'](s,y).item(),.4)
        self.assertAlmostEqual(PROBLEMS['macro-f1']['reference'](torch.tensor([1,1]),torch.tensor([1,1]),3).item(),1/3)

    def test_norms_match_pytorch(self):
        torch.manual_seed(86)
        for id in ('batchnorm1d','batchnorm2d'):
            for name,args in PROBLEMS[id]['cases']():
                x,w,b,rm,rv,*optional=args
                training=optional[0] if optional else True
                momentum=optional[1] if len(optional)>1 else .1
                old_mean,old_var=rm.clone(),rv.clone()
                mean,var=rm.clone(),rv.clone()
                expected=torch.nn.functional.batch_norm(x,mean,var,w,b,training,momentum,1e-5)
                actual,nm,nv=PROBLEMS[id]['reference'](*args)
                torch.testing.assert_close(actual,expected)
                torch.testing.assert_close(nm,mean)
                torch.testing.assert_close(nv,var)
                torch.testing.assert_close(rm,old_mean);torch.testing.assert_close(rv,old_var)
        for id,oracle in [('instancenorm',lambda x,w,b:torch.nn.functional.instance_norm(x,weight=w,bias=b)),
                          ('groupnorm',lambda x,w,b,g:torch.nn.functional.group_norm(x,g,w,b)),
                          ('layernorm-nd',lambda x,w,b,shape:torch.nn.functional.layer_norm(x,shape,w,b))]:
            for name,args in PROBLEMS[id]['cases']():
                torch.testing.assert_close(PROBLEMS[id]['reference'](*args),oracle(*args))

    def test_cached_chunk_decode_matches_full_attention(self):
        torch.manual_seed(22)
        q=torch.randn(2,4,7,3,dtype=torch.float64)
        for id,hkv in [('kv-cache-decode',4),('kv-cache-gqa',2)]:
            k=torch.randn(2,hkv,7,3,dtype=torch.float64)
            v=torch.randn(2,hkv,7,5,dtype=torch.float64)
            expected=torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(4//hkv,1),v.repeat_interleave(4//hkv,1),is_causal=True)
            pk,pv=k[:,:,:0],v[:,:,:0]
            pieces=[];start=0
            for end in (2,5,6,7):
                out,pk,pv=PROBLEMS[id]['reference'](q[:,:,start:end],k[:,:,start:end],v[:,:,start:end],pk,pv)
                pieces.append(out); start=end
            torch.testing.assert_close(torch.cat(pieces,2),expected)
            torch.testing.assert_close(pk,k);torch.testing.assert_close(pv,v)

    def test_sliding_multistep_matches_full_window(self):
        torch.manual_seed(23)
        q=torch.randn(1,2,9,4,dtype=torch.float64);k=torch.randn_like(q);v=torch.randn(1,2,9,3,dtype=torch.float64)
        pos=torch.arange(9)
        mask=(pos[None,:]<=pos[:,None])&(pos[None,:]>pos[:,None]-3)
        expected=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        pk,pv=k[:,:,:0],v[:,:,:0];parts=[];start=0
        for end in (2,6,9):
            out,pk,pv=PROBLEMS['kv-cache-sliding']['reference'](q[:,:,start:end],k[:,:,start:end],v[:,:,start:end],pk,pv,3)
            self.assertLessEqual(pk.shape[2],3)
            parts.append(out);start=end
        torch.testing.assert_close(torch.cat(parts,2),expected)
        torch.testing.assert_close(pk,k[:,:,-3:]);torch.testing.assert_close(pv,v[:,:,-3:])

    def test_wrong_auc_ties_and_cache_offsets_fail(self):
        code='import torch\ndef solve(scores,targets):\n p=scores[targets==1];n=scores[targets==0]\n return (p[:,None]>n[None,:]).double().mean()'
        self.assertEqual(judge('roc-auc',code)['status'],'wrong_answer')
        code=PROBLEMS['kv-cache-decode']['solution'].replace('<= past+torch.arange(t,device=q.device)[:,None]','<= torch.arange(t,device=q.device)[:,None]')
        self.assertEqual(judge('kv-cache-decode',code)['status'],'wrong_answer')

if __name__=='__main__':unittest.main(verbosity=2)
