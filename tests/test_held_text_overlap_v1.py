import unittest

from slm_lab.held_text_overlap_v1 import HeldTextOverlap


def stem(prefix):
    return ' '.join(prefix+str(i) for i in range(90))


class HeldTextContracts(unittest.TestCase):
    def test_actual_copied_problem_with_markup_and_added_context_is_detected(self):
        text=stem('term')
        index=HeldTextOverlap([dict(audit_id='held-a',question=text),dict(audit_id='held-b',question=stem('other'))])
        variant='Extra context.\n\n'+' '.join('**'+word+'**' for word in text.split())+'\nAdditional original discussion.'
        self.assertEqual(index.literal(variant),{})
        self.assertEqual(set(index.lexical_copies(variant)),{'held-a'})
        self.assertEqual(index.lexical_copies(stem('unrelated')), {})
        self.assertEqual(set(index.literal('before\n'+text+'\nafter')), {'held-a'})

    def test_generic_boilerplate_does_not_turn_different_problems_into_copies(self):
        boilerplate=' '.join('shared'+str(i) for i in range(70))
        rows=[dict(audit_id=str(i),question=boilerplate+' '+stem('task'+str(i)+'x')) for i in range(5)]
        index=HeldTextOverlap(rows,maximum_document_frequency=2)
        self.assertEqual(index.lexical_copies(boilerplate+' '+stem('different')), {})
        self.assertEqual(set(index.lexical_copies(rows[0]['question'])), {'0'})
        self.assertEqual(set(index.literal(rows[0]['question'])), {'0'})

    def test_literal_boundaries_duplicates_unicode_and_gold_independence(self):
        rows=[dict(audit_id='a',question='Who'),dict(audit_id='b',question='ＷＨＯ'),
              dict(audit_id='c',question='what is 2 + 2?',answer='unused')]
        index=HeldTextOverlap(rows)
        self.assertEqual(index.literal('whole somewhere'),{})
        self.assertEqual(set(index.literal('WHO? what is 2 + 2?')), {'a','b','c'})
        changed=[{**r,'answer':'changed gold'} for r in rows]
        self.assertEqual(index.literal('who'),HeldTextOverlap(changed).literal('who'))
        with self.assertRaises(ValueError): index.literal('who\0')


if __name__=='__main__':unittest.main()
