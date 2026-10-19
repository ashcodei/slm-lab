"""Literal held stems and declared lexical-copy candidates, without gold labels.

These are source-exposure signals. Generic stem matches are not necessarily the
same problem, and absence is not semantic or complete ancestry independence.
"""
from collections import Counter, defaultdict
import re
import unicodedata

import ahocorasick

WORD = re.compile(r'[^\W_]+(?:_[^\W_]+)*', re.UNICODE)


def normalize(text):
    if not isinstance(text, str) or '\0' in text:
        raise ValueError('Whole original text without boundary sentinel required')
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def shingles(text, width=5):
    words = WORD.findall(normalize(text))
    return {tuple(words[i:i+width]) for i in range(max(0, len(words)-width+1))}


class HeldTextOverlap:
    def __init__(self, rows, *, width=5, maximum_document_frequency=25,
                 minimum_shared=24, minimum_shorter_coverage=.7, minimum_longer_coverage=.3):
        if (not rows or len({r['audit_id'] for r in rows}) != len(rows) or width < 1
                or maximum_document_frequency < 1 or minimum_shared < 1
                or not 0 < minimum_longer_coverage <= minimum_shorter_coverage <= 1):
            raise ValueError('Distinct complete held registry and fixed nonzero thresholds required')
        self.width = width; self.maximum_document_frequency = maximum_document_frequency
        self.minimum_shared = minimum_shared; self.minimum_shorter_coverage = minimum_shorter_coverage
        self.minimum_longer_coverage = minimum_longer_coverage
        self.rows = rows; self.normalized = {r['audit_id']: normalize(r['question']) for r in rows}
        mapping = defaultdict(list)
        for identity, text in self.normalized.items():
            if not text: raise ValueError('Every unchanged held stem must be nonempty')
            mapping[text].append(identity)
        self.automaton = ahocorasick.Automaton()
        for text, identities in mapping.items(): self.automaton.add_word(text, (text, identities))
        self.automaton.make_automaton()
        original = {r['audit_id']: shingles(r['question'], width) for r in rows}
        frequency = Counter(s for values in original.values() for s in values)
        self.uninformative = {s for s, count in frequency.items() if count > maximum_document_frequency}
        self.held = {identity: values - self.uninformative for identity, values in original.items()}
        self.postings = defaultdict(list)
        for identity, values in self.held.items():
            if len(values) >= minimum_shared:
                for value in values: self.postings[value].append(identity)
        self.configuration = dict(algorithm='All literal normalized stems plus exact distinct word-shingle intersections',
            word_shingle_width=width, maximum_held_document_frequency=maximum_document_frequency,
            minimum_distinct_informative_shared_shingles=minimum_shared,
            minimum_shorter_informative_set_coverage=minimum_shorter_coverage,
            minimum_longer_informative_set_coverage=minimum_longer_coverage,
            actual_held_stems=len(rows), actual_unique_literal_patterns=len(mapping),
            actual_informative_indexed_shingles=len(self.postings),
            actual_high_frequency_shingles_ignored_only_for_lexical_copy_check=len(self.uninformative),
            literal_checks_do_not_ignore_high_frequency_stems=True,
            no_gold_or_model_score_used=True, semantic_equivalence_not_certified=True)

    def literal(self, text):
        value = normalize(text); found = {}
        for end, (pattern, identities) in self.automaton.iter(value):
            start = end-len(pattern)+1
            # Do not confuse a short word/stem with a suffix inside another word.
            if (pattern[0].isalnum() or pattern[0] == '_') and start and (value[start-1].isalnum() or value[start-1] == '_'):
                continue
            if (pattern[-1].isalnum() or pattern[-1] == '_') and end+1 < len(value) and (value[end+1].isalnum() or value[end+1] == '_'):
                continue
            for identity in identities:
                found.setdefault(identity, dict(normalized_character_start=start, normalized_character_length=len(pattern)))
        return found

    def lexical_copies(self, text):
        values = shingles(text, self.width)-self.uninformative
        if len(values) < self.minimum_shared: return {}
        candidates = Counter(identity for value in values for identity in self.postings.get(value, ()))
        matches = {}
        for identity, shared in candidates.items():
            if shared < self.minimum_shared: continue
            other = len(self.held[identity]); shorter = min(len(values), other); longer = max(len(values), other)
            a, b = shared/shorter, shared/longer
            if a >= self.minimum_shorter_coverage and b >= self.minimum_longer_coverage:
                matches[identity] = dict(distinct_shared_informative_shingles=shared,
                    candidate_informative_shingles=len(values), held_informative_shingles=other,
                    shorter_set_coverage=a, longer_set_coverage=b)
        return matches
