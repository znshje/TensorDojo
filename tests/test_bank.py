import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
import torch
torch.set_num_threads(1)
from bank import BankStore,BankError,validate_bundle,encode,decode,export_problem,executable,apply_overlay
from catalog import BUILTIN_PROBLEMS
from server import prepare_import
ROOT=Path(__file__).resolve().parents[1]

class BankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=json.loads((ROOT/'examples/question-bank.json').read_text())
        cls.rows=prepare_import(validate_bundle(cls.bundle))
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'bank.json'
        self.store=BankStore(BUILTIN_PROBLEMS,self.path)
    def test_create_update_delete_restore_persist(self):
        self.store.import_prepared(self.rows,0)
        self.assertEqual(self.store.info()['revision'],1)
        self.assertEqual(len(self.store.snapshot()),44)
        id=self.rows[0]['spec']['id']
        with self.assertRaises(BankError):self.store.import_prepared(self.rows,0,True)
        with self.assertRaises(BankError):self.store.import_prepared(self.rows,1)
        self.store.import_prepared(self.rows,1,True)
        self.store.delete(id,2)
        self.assertNotIn(id,self.store.snapshot())
        self.assertIn(id,self.store.info()['deleted'])
        self.store.restore(id,3)
        reloaded=BankStore(BUILTIN_PROBLEMS,self.path)
        self.assertIn(id,reloaded.snapshot())
        self.assertEqual(reloaded.info()['revision'],4)
        effective=dict(BUILTIN_PROBLEMS);apply_overlay(effective,self.path)
        self.assertEqual(effective[id]['_public']['totalTests'],3)
    def test_builtin_delete_restore(self):
        self.store.delete('softmax',0)
        self.assertNotIn('softmax',BankStore(BUILTIN_PROBLEMS,self.path).snapshot())
        self.store.restore('softmax',1)
        self.assertIn('softmax',self.store.snapshot())
    def test_dry_run_does_not_write(self):
        result=self.store.import_prepared(self.rows,0,dry_run=True)
        self.assertTrue(result['dry_run']);self.assertFalse(self.path.exists())
        self.assertEqual(self.store.info()['revision'],0)
    def test_export_roundtrip_all_builtins(self):
        bundle=self.store.export()
        validated=validate_bundle(bundle)
        self.assertEqual(len(validated),43)
        # Validate real expected outputs, code restrictions and gradients for all exports.
        rows=prepare_import(validated)
        self.store.import_prepared(rows,0,overwrite=True)
        exported=self.store.export()
        self.assertEqual(bundle['problems'],exported['problems'])
        self.assertEqual(len(rows),43)
    def test_reject_bad_schema_and_expected_output(self):
        for edit in [lambda p:p.update(id='../bad'),lambda p:p.update(minutes=-1),lambda p:p.update(difficulty='unknown'),lambda p:p.update(signature='x=missing()'),lambda p:p['tests'][0]['args'][0].update(shape=[999999999])]:
            bad=copy.deepcopy(self.bundle);edit(bad['problems'][0])
            with self.assertRaises(BankError):validate_bundle(bad)
        bad=copy.deepcopy(self.bundle);bad['problems'][0]['tests'][0]['expected']['data'][0]=999
        with self.assertRaises(BankError):prepare_import(validate_bundle(bad))
        self.assertFalse(self.path.exists())
    def test_duplicate_batch_rejected(self):
        bad=copy.deepcopy(self.bundle);bad['problems']*=2
        with self.assertRaises(BankError):validate_bundle(bad)
    def test_codec_empty_tuple_and_scalars(self):
        values=(torch.zeros(2,0,3,dtype=torch.float64),torch.tensor(2.),(4,True,None))
        actual=decode(encode(values));torch.testing.assert_close(actual,values)

if __name__=='__main__':unittest.main(verbosity=2)
