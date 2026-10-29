"""Guard prompt provenance, answer leakage, balanced pilots, and safe resume."""
import copy
import unittest
from robustness_protocol import ROOT,read,validate_suite,select_rows,validate_prefix,valid_name,expected_input


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.suite=read(ROOT/'study/robustness/suite.json')
    def test_approved_existing_suite_and_balanced_pilot(self):
        validate_suite(self.suite)
        rows=select_rows(self.suite,5)
        self.assertEqual(len(rows),40)
        for domain in ('forget','retain'):
            self.assertEqual(len([r for r in rows if r['domain']==domain]),20)
        self.assertEqual(len(select_rows(self.suite,0)),320)
    def test_answer_added_to_prompt_is_rejected(self):
        bad=copy.deepcopy(self.suite)
        bad['items'][1]['prompt']+=' '+bad['items'][1]['reference_answer']
        with self.assertRaises(ValueError):validate_suite(bad)
    def test_changed_question_target_id_or_source_is_rejected(self):
        for key in ('question','reference_answer','id'):
            bad=copy.deepcopy(self.suite);bad['items'][0][key]='changed'
            with self.assertRaises(ValueError):validate_suite(bad)
        bad=copy.deepcopy(self.suite);bad['source_sha256']='changed'
        with self.assertRaises(ValueError):validate_suite(bad)
    def test_resume_rejects_holes_duplicates_and_partial_batches(self):
        rows=self.suite['items']
        validate_prefix(rows[:8],rows,8)
        validate_prefix(rows,rows,8)
        for bad in (rows[:7],rows[1:9],[rows[0]]*8):
            with self.assertRaises(ValueError):validate_prefix(bad,rows,8)
    def test_names_and_limits_cannot_escape_or_claim_wrong_scope(self):
        for name in ('../old','a/b','','..'):
            with self.assertRaises(ValueError):valid_name(name)
        for limit in (-1,41):
            with self.assertRaises(ValueError):select_rows(self.suite,limit)
    def test_wrapped_processed_input_changes_only_source_question(self):
        rows=self.suite['items'][:4]
        direct=expected_input(rows[0])
        for row in rows[1:]:
            self.assertEqual(expected_input(row),direct.replace(row['question'],row['prompt'],1))


if __name__=='__main__':unittest.main()
