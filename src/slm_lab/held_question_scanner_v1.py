"""Conservative normalized question-substring checks over native input text.

These checks cannot establish semantic independence, complete ancestral token
consumption, or absence of unannotated image-text exposure.
"""
import unicodedata

import ahocorasick


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


class QuestionScanner:
    def __init__(self, questions):
        mapping = {}
        for row in questions:
            text = normalize(row['question'])
            if not text or '\0' in text: raise ValueError('Nonempty original question without record-boundary sentinel required')
            mapping.setdefault(text, []).append(row['audit_id'])
        if not mapping: raise ValueError('Original held questions required')
        self.automaton = ahocorasick.Automaton()
        for text, identities in mapping.items(): self.automaton.add_word(text, (len(text), identities))
        self.automaton.make_automaton()
        self.matches = {}; self.maximum_normalized_characters = max(map(len, mapping))

    def text(self, value, source):
        normalized = normalize(value)
        for end, (length, identities) in self.automaton.iter(normalized):
            for identity in identities:
                self.matches.setdefault(identity, dict(source=source, normalized_character_start=end-length+1))

    def native(self, tokenizer, ids, source):
        text = tokenizer.decode(ids, skip_special_tokens=False)
        for marker in ['<bos>','<eos>','<pad>']: text = text.replace(marker, '\0')
        self.text(text, source)
