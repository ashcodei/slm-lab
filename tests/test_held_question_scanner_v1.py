from slm_lab.held_question_scanner_v1 import QuestionScanner


def test_exact_unicode_whitespace_matches_keep_alias_ids_and_do_not_cross_record_boundaries():
    scanner=QuestionScanner([dict(audit_id='a',question='Straße Å?'),dict(audit_id='b',question='strasse Å?'),
        dict(audit_id='c',question='alpha beta')])
    scanner.text('prefix STRASSE\n A\u030a? suffix',dict(family='train',position=3))
    assert set(scanner.matches)=={'a','b'}
    scanner.text('alpha\0 beta',dict(family='train',position=4))
    assert 'c' not in scanner.matches
    scanner.text('alpha  beta',dict(family='train',position=5))
    assert scanner.matches['c']['source']['position']==5


def test_special_token_boundaries_do_not_fuse_unrelated_training_examples():
    class Tokenizer:
        def decode(self,ids,skip_special_tokens):return 'alpha<eos><bos> beta'
    scanner=QuestionScanner([dict(audit_id='a',question='alpha beta')])
    scanner.native(Tokenizer(),[1,3,2,1,4,2],dict(family='train'))
    assert not scanner.matches
