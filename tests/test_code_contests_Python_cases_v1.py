from pathlib import Path
import unittest

from slm_lab.code_contests_Python_cases_v1 import case_contract,execute_reference,token_equal


class OriginalCaseExecution(unittest.TestCase):
    def test_actual_full_standard_program_reads_buffer_main_guard_and_all_three_case_kinds(self):
        program='import sys, math\nclass Solver:\n    def run(self):\n        print(2*int(sys.stdin.buffer.readline()))\nif __name__=="__main__":\n    Solver().run()\n'
        cases=[dict(kind=kind,position=0,input=str(n)+'\n',expected=str(n*2)+'\n')
               for kind,n in [('public_tests',3),('private_tests',7),('generated_tests',-4)]]
        result=execute_reference(program,cases)
        self.assertTrue(result['all_cases_observed']);self.assertTrue(result['all_original_finite_cases_pass'])
        self.assertEqual(result['actual_observed_case_count'],3)
        self.assertEqual(result['counts_by_original_kind'],dict(public_tests=1,private_tests=1,generated_tests=1))
        self.assertTrue(all(r['actual_output_tokens_equal_expected'] for r in result['original_case_verdicts']))

    def test_expected_values_are_outside_fresh_candidate_filesystem_and_broadening_does_not_hide_mismatch(self):
        program='import os\nprint(" ".join(sorted(os.listdir("."))))\n'
        cases=[dict(kind='public_tests',position=0,input='only-input\n',expected='program.py\n')]
        result=execute_reference(program,cases)
        self.assertTrue(result['all_original_finite_cases_pass'])
        self.assertTrue(result['expected_outputs_never_written_to_candidate_readable_filesystem'])
        self.assertTrue(token_equal(b'1  2\n','1\t2\n'));self.assertFalse(token_equal(b'12\n','1 2\n'))
        cases.append(dict(kind='private_tests',position=0,input='second\n',expected='hidden wrong output\n'))
        result=execute_reference(program,cases)
        self.assertFalse(result['all_original_finite_cases_pass']);self.assertTrue(result['all_cases_observed'])
        self.assertEqual(result['total_cases'],2);self.assertEqual(result['first_unqualified_original_case_ordinal'],1)
        cases.append(dict(kind='generated_tests',position=0,input='third\n',expected='program.py\n'))
        result=execute_reference(program,cases)
        self.assertFalse(result['all_cases_observed']);self.assertEqual(result['actual_observed_case_count'],2)
        self.assertEqual(result['total_cases'],3)

    def test_infinite_execution_and_incomplete_case_population_cannot_earn_full_case_success(self):
        cases=[dict(kind='public_tests',position=0,input='1\n',expected='2\n')]
        result=execute_reference('while True: pass\n',cases,seconds_per_case=.2,seconds_total=1.)
        self.assertFalse(result['all_original_finite_cases_pass']);self.assertFalse(result['all_cases_observed'])
        self.assertEqual(result['original_case_verdicts'][0]['process_status'],'wall_deadline_exceeded')
        with self.assertRaises(ValueError):case_contract([dict(cases[0],position=1)])
        with self.assertRaises(ValueError):case_contract([dict(cases[0],input='x'*(4*1024**2+1))])


if __name__=='__main__':unittest.main()
