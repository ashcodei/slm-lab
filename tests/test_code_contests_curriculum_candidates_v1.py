import unittest
from types import SimpleNamespace

from slm_lab.code_contests_curriculum_candidates_v1 import (native_sequence,
    normalized_program_hash, original_problem_keys, prompt, select_complete_solutions, url_key)


class BytesTokenizer:
    def encode(self, value): return SimpleNamespace(ids=[byte + 4 for byte in value.encode()])


class Problem:
    description = 'Read one integer and print twice its value.\nKeep all details.'
    input_file = ''; output_file = ''
    private_tests = [SimpleNamespace(input='SECRET_INPUT', output='SECRET_EXPECTED')]
    generated_tests = [SimpleNamespace(input='GENERATED_INPUT', output='GENERATED_EXPECTED')]
    incorrect_solutions = [SimpleNamespace(language=3, solution='print(999999)')]

    def __init__(self, values): self.solutions = values
    def HasField(self, name): return bool(getattr(self, name))


class CandidateContracts(unittest.TestCase):
    def test_existing_held_problem_identity_survives_alternative_urls(self):
        self.assertEqual(url_key('https://codeforces.com/problemset/problem/1101/B?locale=en'),
                         url_key('https://www.codeforces.com/contest/1101/problem/B/'))
        self.assertNotEqual(url_key('https://codeforces.com/gym/1101/problem/B'),
                            url_key('https://codeforces.com/contest/1101/problem/B'))
        self.assertEqual(url_key('https://www.codechef.com/COOK68/problems/ALTARAY'), ('CODECHEF', 'ALTARAY'))
        self.assertEqual(original_problem_keys(dict(original_source='ATCODER', original_name='p02537 original title',
            original_CF_contest_id=None, original_CF_problem_index=None)), [('CODENET', 'p02537')])

    def test_whole_programs_no_private_answers_no_python2_relabelling(self):
        good = 'n=int(input())\nprint(2*n)\n'
        py2 = 'n=input()\nprint 2*n\n'
        p = Problem([SimpleNamespace(language=3, solution=good), SimpleNamespace(language=1, solution=py2),
            SimpleNamespace(language=3, solution=good), SimpleNamespace(language=3, solution='x'*5000)])
        selected, counts, matches = select_complete_solutions(p, BytesTokenizer(),
            {normalized_program_hash(good): ['reserved-fixture']}, maximum_per_language=1)
        self.assertEqual([r['response'] for r in selected], [good, py2])
        self.assertEqual([r['language'] for r in selected], ['Python3', 'Python2'])
        self.assertEqual(len(matches), 2)
        for row in selected:
            self.assertNotIn('SECRET_', row['prompt']); self.assertNotIn('GENERATED_', row['prompt'])
            self.assertNotIn('999999', row['response'])
            start = next(i for i, v in enumerate(row['labels']) if v != -100)
            self.assertEqual(row['ids'][-1], 2); self.assertEqual(row['labels'][start:], row['ids'][start:])
            self.assertEqual(bytes(v-4 for v in row['ids'][start:-1]).decode(), row['response'])
        self.assertEqual(counts['original_publisher_correct_solutions_visited'], 4)

    def test_context_is_rejected_whole_without_clipping_original_source(self):
        code = 'a'*5000
        sequence, reason = native_sequence(BytesTokenizer(), 'a problem', code)
        self.assertIsNone(sequence); self.assertEqual(reason, 'complete_context_exceeds_4096')
        self.assertEqual(len(code), 5000)
        p=Problem([]); p.input_file='data.in'; p.output_file='data.out'
        text=prompt(p, 3)
        self.assertIn("'data.in'", text); self.assertTrue(text.endswith(p.description))


if __name__ == '__main__': unittest.main()
