"""Exercise actual grammar byte ranges before transforming public code prompts."""
import json
from pathlib import Path

from slm_lab.diverse_images import atomic_json
from slm_lab.goruby_code_headers import HeaderParser

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = HeaderParser(ROOT)
    cases = [
        ('go-ordinary', 'Go', 'func sum(a int, b int) int { return a + b }', False, False, 'func sum(a int, b int) int '),
        ('go-interface', 'Go', 'func inspect(value interface{}) interface{} { return value }', True, False,
         'func inspect(value interface{}) interface{} '),
        ('go-result-struct', 'Go', 'func create() struct { Name string } { return struct { Name string }{} }', True, False,
         'func create() struct { Name string } '),
        ('go-receiver', 'Go', 'func (r *Reader) Read(value interface{}) error { return nil }', True, False,
         'func (r *Reader) Read(value interface{}) error '),
        ('go-unicode', 'Go', 'func 数(value interface{}) interface{} { return value }', True, False,
         'func 数(value interface{}) interface{} '),
        ('ruby-body', 'Ruby', 'def sum(a, b)\n a + b\nend', False, True, 'def sum(a, b)'),
        ('ruby-default', 'Ruby', 'def configure(options = {})\n options\nend', True, False, 'def configure(options = {})'),
        ('ruby-keyword', 'Ruby', 'def configure(options: {})\n options\nend', True, False, 'def configure(options: {})'),
        ('ruby-singleton', 'Ruby', 'def self.send(value)\n value\nend', False, True, 'def self.send(value)'),
        ('ruby-multiline', 'Ruby', 'def send(\n value,\n options = {}\n)\n value\nend', True, False,
         'def send(\n value,\n options = {}\n)'),
        ('ruby-empty', 'Ruby', 'def nothing\nend', False, False, 'def nothing'),
        ('ruby-unicode', 'Ruby', 'def 名前(value = {})\n value\nend', True, False, 'def 名前(value = {})'),
    ]
    results = []
    for name, language, source, cut, copied, header in cases:
        measured = parser.measure(language, source)
        if (not measured['parsed'] or measured['header_text_cut'] != cut
                or measured['method_body_text_in_context'] != copied or measured['complete_header'] != header):
            raise ValueError('Actual header grammar probe failed: ' + name + ': ' + json.dumps(measured))
        results.append({'name': name, **measured})
    invalid = parser.measure('Go', 'func broken(value {')
    if invalid['parsed']: raise ValueError('Incomplete Go grammar input was accepted')
    atomic_json(ROOT / 'reports/goruby-header-parser-probes-v1.json', {'passed': True, 'examples': len(results),
        'results': results, 'incomplete_source_rejected': True, 'versions': parser.metadata['versions'],
        'toolchain_manifest_sha256': parser.manifest_sha, 'source_programs_executed': False, 'goal_achieved': False})
    print('Actual Go/Ruby grammar header probes passed', len(results))


if __name__ == '__main__': main()
