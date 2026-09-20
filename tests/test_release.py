import json
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sleepinn_study.review import ReviewSession
from sleepinn_study.workflow import assemble_final, answer_prompt, apply_ai_reviews
from sleepinn_study.providers import request_json
from sleepinn_study.knowledge_judge import request
from sleepinn_rag.september_core import fuse_lists, strategy_matrix
from sleepinn_study.audit import verify_release
from sleepinn_study.io import read_jsonl, read_json, digest


class ReleaseTests(unittest.TestCase):
    def test_ai_patch_preserves_input_and_cannot_forge_review(self):
        items = [dict(item_id='x', item_type='free_text_qa', question='Question?', answer='Original')]
        original = deepcopy(items)
        changed = apply_ai_reviews(items, [dict(item_id='x',decision='revise',patch={'answer':'Correction'})])
        self.assertEqual(items, original)
        self.assertEqual(changed[0]['answer'], 'Correction')
        self.assertTrue(changed[0]['metadata']['ai_review']['human_review_required'])
        with self.assertRaises(ValueError):
            apply_ai_reviews(items,[dict(item_id='x',decision='revise',patch={'metadata':{'human_review':'accepted'}})])

    def test_full_review_required(self):
        with tempfile.TemporaryDirectory() as folder:
            items = [dict(item_id=str(i), item_type='free_text_qa', question='Question?', answer='Reference', options=None) for i in range(2)]
            session = ReviewSession(items, folder)
            session.save('0', items[0], 'accepted', 'reviewer', True)
            with self.assertRaisesRegex(ValueError, 'still pending'):
                session.export()
            session.save('1', items[1], 'rejected', 'reviewer', False, 'Ambiguous reference')
            exported, count = session.export()
            self.assertEqual(count, 1)
            self.assertTrue(read_json(exported.with_suffix('.manifest.json'))['all_items_decided'])
            clinical = [dict(item_id='c', item_type='clinical_case', question='Case?', answer='Reference')]
            other = ReviewSession(clinical, Path(folder) / 'clinical')
            other.save('c', clinical[0], 'accepted', 'reviewer', True)
            clinical_export, _ = other.export()
            final = assemble_final(exported, clinical_export, Path(folder) / 'final.jsonl')
            self.assertEqual(len(read_jsonl(final)), 2)
            exported.write_text('tampered', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                assemble_final(exported, clinical_export, Path(folder) / 'another.jsonl')

    def test_balanced_merge_and_rrf(self):
        merged, _ = fuse_lists([[1,2,3]], [[3,4,1]], {'rrf':False,'interleave':True})
        self.assertEqual(merged, [1,3,2,4])
        merged, scores = fuse_lists([[1,2]], [[2,3]], {'rrf':True})
        self.assertEqual(merged[0], 2)
        self.assertAlmostEqual(scores[2], 1/62 + 1/61)
        matrix = strategy_matrix(ROOT / 'src')
        selected = next(r for r in matrix if r['strategy']=='scope_hybrid_rrf_rerank_jiwar')
        self.assertTrue(selected['scope'] and selected['rerank'] and selected['jiwar'])
        self.assertFalse(selected['mq'])

    def test_no_rag_prompt_hashes_match_historical_prompts(self):
        for domain in ['knowledge', 'clinical']:
            hashes = {r['item_id']:r['prompt_sha256']['no_rag'] for r in read_jsonl(ROOT / f'data/config/{domain}_prompt_hashes.jsonl')}
            for item in read_jsonl(ROOT / f'data/final/{domain}.jsonl'):
                self.assertEqual(digest(answer_prompt(item).encode()), hashes[item['item_id']])

    def test_knowledge_judge_drops_identity(self):
        job = dict(blind_id='x', question='Q', item_type='free_text_qa', candidate_reference='A', response='A', model_id='SECRET_MODEL', precision='SECRET_PRECISION', mode='SECRET_MODE')
        payload = json.dumps(request(job, {'generationConfig':{}}, {}))
        for key in ['SECRET_MODEL','SECRET_PRECISION','SECRET_MODE']:
            self.assertNotIn(key, payload)

    def test_api_output_boundary_precedes_network(self):
        with patch('urllib.request.urlopen') as network:
            with self.assertRaisesRegex(ValueError, 'ignored outputs'):
                request_json('gemini','unused',{},ROOT / 'data/unsafe',enabled=True)
            network.assert_not_called()

    def test_published_populations_and_hashes(self):
        self.assertEqual(verify_release()['clinical_systems'],19)


if __name__ == '__main__':
    unittest.main()
