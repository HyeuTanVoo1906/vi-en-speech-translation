"""CPU tests for leakage boundaries, gate policy, and calibration isolation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from argparse import Namespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asr.context_gate import features, label, choose, FEATURES
from context_gate_pipeline import partition, train, evaluate, validate_pairs, immutable_jsonl


class GateTests(unittest.TestCase):
    def test_features_reference_blind_and_labels(self):
        import inspect
        self.assertNotIn('reference', inspect.signature(features).parameters)
        x = features('a b', 'a c', 'c', -2, -1, -1, 10)
        self.assertEqual(len(x), len(FEATURES))
        self.assertEqual(label('a c','a b','a c')['benefit'], 1)
        self.assertEqual(label('a b','a b','a c')['harm'], 1)
        self.assertEqual(label('a b','a c','a d')['benefit'], 0)

    def test_truncation_and_ties_keep(self):
        model = dict(benefit={'constant':1},harm={'constant':0},benefit_threshold=.5,harm_threshold=.1)
        r = dict(base='a',context='b',features=[0]*len(FEATURES),context_truncated=False,history_truncated=False)
        self.assertEqual(choose(model,r)[0],'REPLACE')
        self.assertEqual(choose(model,r|{'context_truncated':True})[0],'KEEP')
        self.assertEqual(choose(model,r|{'history_truncated':True})[0],'KEEP')
        self.assertEqual(choose(model,r|{'context':'A!'})[0],'KEEP')

    def fixture(self, directory):
        rows = []
        for p in ('train','cal','dev'):
            for i in range(30):
                benefit = i % 2 == 0
                ref, base, context = 'a b', 'a c' if benefit else 'a b', 'a b' if benefit else 'a c'
                row = dict(id=f'{p}-{i}',video_id=f'{p}-video',partition=p,base=base,context=context,
                           context_truncated=False,history_truncated=False,
                           features=features(base,context,'b',-2,-1 if benefit else -3,-1,10),
                           **label(ref,base,context))
                rows.append(row)
        immutable_jsonl(directory/'segments.jsonl', rows)
        immutable_jsonl(directory/'pairs.jsonl', rows)
        (directory/'generation.json').write_text('{}')
        return rows

    def test_end_to_end_and_dev_does_not_tune(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            a,b = Path(a),Path(b)
            rows = self.fixture(a)
            self.fixture(b)
            altered = [r if r['partition']!='dev' else r|{'benefit':1-r['benefit'],'harm':1-r['harm']} for r in rows]
            (b/'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in altered))
            for path in (a,b):
                train(Namespace(run_dir=path,allow_partial=False,max_harm_rate=.01))
            ma,mb = [json.loads((p/'gate.json').read_text()) for p in (a,b)]
            for field in ('benefit','harm','benefit_threshold','harm_threshold','calibration'):
                self.assertEqual(ma[field],mb[field])
            self.assertLess(ma['calibration']['gate_wer'],ma['calibration']['baseline_wer'])
            self.assertEqual(ma['calibration']['harmed'],0)
            evaluate(Namespace(run_dir=a))
            self.assertTrue((a/'evaluation.json').exists())
            with self.assertRaisesRegex(ValueError, 'already frozen'):
                train(Namespace(run_dir=a,allow_partial=False,max_harm_rate=.01))

    def test_partial_generation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            rows=self.fixture(d)
            (d/'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1]))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                validate_pairs(d)

    def test_video_partition_stable(self):
        self.assertEqual(partition('abc'),partition('abc'))
        self.assertEqual({partition(str(i)) for i in range(100)}, {'train','cal','dev'})

    def test_cross_video_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            rows=self.fixture(d)
            rows=[r|{'video_id':'same-video'} for r in rows]
            for name in ('segments.jsonl','pairs.jsonl'):
                (d/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'video leakage'):
                validate_pairs(d)

    def test_no_benefit_refuses_training(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            rows=self.fixture(d)
            rows=[r|{'benefit':0} for r in rows]
            (d/'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'No beneficial'):
                train(Namespace(run_dir=d,allow_partial=False,max_harm_rate=.01))
            self.assertFalse((d/'gate.json').exists())


if __name__=='__main__':
    unittest.main()
