import json
from pathlib import Path
import tempfile
import unittest

from slm_lab.source_jsonl_records_v1 import records


class SourceJSONLContracts(unittest.TestCase):
    def test_unicode_line_separators_inside_original_question_are_not_records(self):
        text='first\x85second\u2028third\u2029tail'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'original.jsonl'
            path.write_text(json.dumps(dict(id='original',question=text),ensure_ascii=False)+'\n'+
                json.dumps(dict(id='second',question='a\nb'))+'\n')
            self.assertEqual([r['question'] for r in records(path)],[text,'a\nb'])
            self.assertGreater(len(path.read_text().splitlines()),2)

    def test_incomplete_ambiguous_and_oversized_records_are_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'original.jsonl'
            for text in ['{"question":"unfinished','{"id":1,"id":2}\n','{"value":NaN}\n','\n']:
                path.write_text(text)
                with self.assertRaises(ValueError):list(records(path))
            path.write_text('{"id":1}')
            self.assertEqual(list(records(path)),[dict(id=1)])
            with self.assertRaises(ValueError):list(records(path,maximum_record_characters=4))


if __name__=='__main__':unittest.main()
