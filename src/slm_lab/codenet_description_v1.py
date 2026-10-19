"""Retain visible original problem text and conservatively identify sample I/O."""
from collections import defaultdict
from html.parser import HTMLParser
import re
import unicodedata


def sample_heading(text):
    # Normalize heading labels only; full-width characters in sample data matter.
    text = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    output = r'^(?:sample\s+output|output\s+(?:for\s+(?:the\s+)?sample\s+input|example|sample)|出力例|サンプル出力|예제\s*출력)'
    input_ = r'^(?:sample\s+input|input\s+(?:example|sample)|入力例|サンプル入力|예제\s*입력)'
    kind = 'output' if re.match(output, text) else 'input' if re.match(input_, text) else None
    number = re.search(r'([0-9]+)\s*[.:)]?\s*$', text)
    return (kind, number.group(1) if number else None) if kind else None


class Description(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.pre = None
        self.heading = None
        self.active_sample = None
        self.sample_pre_seen = False
        self.samples = []
        self.images = []
        self.embedded = []
        self.suppressed = []
        self.skip = []
        self.math_script = False

    def visible(self, text):
        self.parts.append(text)
        if self.pre is not None:
            self.pre.append(text)
        if self.heading is not None:
            self.heading.append(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if self.skip:
            if tag in {'script', 'style'}:
                self.skip.append(tag)
            return
        if tag == 'script' and attrs.get('type', '').lower().startswith('math/tex'):
            self.math_script = True
            self.visible('$')
        elif tag in {'script', 'style'}:
            self.skip.append(tag)
            self.suppressed.append(tag)
        elif re.fullmatch(r'h[1-6]', tag):
            self.visible('\n\n')
            self.heading = []
            self.active_sample = None
            self.sample_pre_seen = False
        elif tag == 'pre':
            self.parts.append('\n')
            self.pre = []
        elif tag == 'br':
            self.visible('\n')
        elif tag in {'p', 'div', 'section', 'article', 'ul', 'ol', 'li', 'tr'}:
            self.visible('\n')
        elif tag in {'td', 'th'}:
            self.visible('\t')
        elif tag == 'sup':
            self.visible('^{')
        elif tag == 'sub':
            self.visible('_{')
        elif tag == 'img':
            self.images.append({'src': attrs.get('src'), 'alt': attrs.get('alt')})
            self.visible('[Image' + (': ' + attrs['alt'] if attrs.get('alt') else '') + ']')
        elif tag in {'svg', 'canvas', 'iframe', 'object', 'embed', 'video', 'audio'}:
            self.embedded.append(tag)

    def handle_endtag(self, tag):
        if self.skip:
            if tag == self.skip[-1]:
                self.skip.pop()
            return
        if tag == 'script' and self.math_script:
            self.visible('$')
            self.math_script = False
        elif re.fullmatch(r'h[1-6]', tag) and self.heading is not None:
            self.active_sample = sample_heading(''.join(self.heading))
            self.heading = None
            self.visible('\n')
        elif tag == 'pre' and self.pre is not None:
            if self.active_sample is not None:
                value = ''.join(self.pre).replace('\r\n', '\n').replace('\r', '\n')
                # HTML block formatting contributes one boundary newline, not data.
                if value.startswith('\n'):
                    value = value[1:]
                if value.endswith('\n'):
                    value = value[:-1]
                self.samples.append((*self.active_sample, value))
                self.sample_pre_seen = True
            self.pre = None
            self.parts.append('\n')
        elif tag in {'sup', 'sub'}:
            self.visible('}')
        elif tag in {'p', 'div', 'section', 'article', 'ul', 'ol', 'li', 'tr'}:
            self.visible('\n')

    def handle_data(self, data):
        if not self.skip:
            # Explanation prose after a labelled sample can have its own pre block.
            # It is original context, not another output for the preceding label.
            if self.pre is None and self.heading is None and self.sample_pre_seen and data.strip():
                self.active_sample = None
            self.visible(data)


def parse_description(html):
    parser = Description()
    parser.feed(html)
    parser.close()
    by_number = defaultdict(lambda: defaultdict(list))
    for kind, number, value in parser.samples:
        # Identical bilingual sample blocks are one original example.
        if value not in by_number[number][kind]:
            by_number[number][kind].append(value)
    cases = []
    issues = []
    for number, pair in by_number.items():
        if len(pair['input']) != 1 or len(pair['output']) != 1:
            issues.append({'sample_number': number, 'inputs': len(pair['input']),
                           'outputs': len(pair['output']), 'reason': 'Unpaired/ambiguous sample blocks'})
            continue
        if not pair['input'][0] and not pair['output'][0]:
            issues.append({'sample_number': number, 'reason': 'Both sample blocks empty'})
            continue
        cases.append({'number': number, 'input': pair['input'][0], 'output': pair['output'][0]})
    if parser.pre is not None or parser.heading is not None or parser.math_script or parser.skip:
        issues.append({'reason': 'Unclosed pre/heading/math/script/style block'})
    return {'text': ''.join(parser.parts).strip(), 'samples': cases,
        'sample_parse_issues': issues, 'recognized_samples_usable': bool(cases) and not issues,
        'image_references': parser.images, 'embedded_media_tags': parser.embedded,
        'requires_media_review': bool(parser.images or parser.embedded),
        'suppressed_script_style_blocks': len(parser.suppressed),
        'sample_scope': 'Recognized explicit sample headings only; not all test cases, a certified complete problem, or a correctness/efficiency proof.'}
