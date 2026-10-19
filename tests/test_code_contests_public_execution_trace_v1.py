import base64
import hashlib
from types import SimpleNamespace
import unittest

from slm_lab.code_contests_public_execution_trace_v1 import public_trace


class BytesTokenizer:
    def encode(self,text):return SimpleNamespace(ids=[v+4 for v in text.encode()])


def fixture():
    code='n=int(input())\nprint(2*n)\n';sha=lambda text:hashlib.sha256(text.encode()).hexdigest()
    row=dict(id='source:positive:0',source_problem_id='original-source',group_id='group',split='train',publisher_split='train',
        language='Python3',response=code,original_program_sha256=sha(code),original_solution_position=0)
    problem=SimpleNamespace(public_tests=[SimpleNamespace(input='7\n',output='14\n')],
        private_tests=[SimpleNamespace(input='PRIVATE_INPUT_DO_NOT_EXPOSE',output='PRIVATE_EXPECTED_DO_NOT_EXPOSE')],
        generated_tests=[SimpleNamespace(input='GENERATED_INPUT_DO_NOT_EXPOSE',output='GENERATED_EXPECTED_DO_NOT_EXPOSE')])
    public=dict(kind='public_tests',original_kind_position=0,process_status='completed',original_input_sha256=sha('7\n'),
        original_expected_sha256=sha('14\n'),actual_stdout_b64=base64.b64encode(b'14\n').decode())
    ref=dict(original_problem_id='original-source',original_program_sha256=sha(code),original_solution_position=0,all_original_finite_cases_pass=True,
        actual_execution=dict(all_cases_observed=True,all_original_finite_cases_pass=True,
            counts_by_original_kind=dict(public_tests=1,private_tests=1,generated_tests=1),actual_observed_case_count=3,
            original_case_verdicts=[public,dict(kind='private_tests'),dict(kind='generated_tests')]))
    return row,problem,ref


class PublicTraceContracts(unittest.TestCase):
    def test_actual_stdout_target_full_program_and_only_original_public_input(self):
        row,problem,ref=fixture();trace,disposition=public_trace(row,problem,ref,BytesTokenizer())
        self.assertEqual(trace['response'],'14\n');self.assertIn(row['response'],trace['prompt']);self.assertIn('7\n',trace['prompt'])
        self.assertNotIn('PRIVATE_',trace['prompt']+trace['response']);self.assertNotIn('GENERATED_',trace['prompt']+trace['response'])
        self.assertTrue(trace['no_additional_neural_parameters_or_teacher']);self.assertEqual(trace['ids'][-1],2)
        start=next(i for i,v in enumerate(trace['labels']) if v!=-100)
        self.assertEqual(bytes(v-4 for v in trace['ids'][start:-1]),b'14\n')

    def test_source_role_wrong_stdout_and_incomplete_full_test_qualification_reject(self):
        row,problem,ref=fixture()
        with self.assertRaises(ValueError):public_trace(dict(row,split='validation'),problem,ref,BytesTokenizer())
        with self.assertRaises(ValueError):public_trace(row,problem,dict(ref,original_problem_id='another-source'),BytesTokenizer())
        ref['actual_execution']['original_case_verdicts'][0]['actual_stdout_b64']=base64.b64encode(b'999\n').decode()
        with self.assertRaises(ValueError):public_trace(row,problem,ref,BytesTokenizer())
        row,problem,ref=fixture();ref['actual_execution']['actual_observed_case_count']=1
        with self.assertRaises(ValueError):public_trace(row,problem,ref,BytesTokenizer())

    def test_whole_context_is_dispositioned_without_clipping(self):
        row,problem,ref=fixture();problem.public_tests[0].input='7\n'+' '*5000
        ref['actual_execution']['original_case_verdicts'][0]['original_input_sha256']=hashlib.sha256(problem.public_tests[0].input.encode()).hexdigest()
        trace,disposition=public_trace(row,problem,ref,BytesTokenizer())
        self.assertIsNone(trace);self.assertEqual(disposition['public_case_dispositions'][0]['reason'],'complete_context_exceeds_4096')
        self.assertEqual(len(problem.public_tests[0].input),5002)


if __name__=='__main__':unittest.main()
