"""Complete publisher-code targets; conservative known-source separation.

Only lossless targets from original train solutions are selected. This module
does not provide execution rewards, infer Python2/Python3 compatibility, or
admit a dataset to optimization. Private/generated test fields are never read
by prompt construction.
"""
import hashlib
import re
import unicodedata
from urllib.parse import unquote, urlsplit


LANGUAGES = {1: ('Python 2', 'Python2'), 2: ('C++', 'C++'),
             3: ('Python 3', 'Python3'), 4: ('Java', 'Java')}


def normalized_question(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def question_hash(text):
    return hashlib.sha256(normalized_question(text).encode()).hexdigest()


def normalized_program_hash(text):
    # A conservative exact-normalized screen. Case remains significant. This
    # can merge whitespace inside literals; a match only quarantines a source.
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC', text).split()).encode()).hexdigest()


def url_key(url):
    if not url: return None
    parsed = urlsplit(url)
    host = parsed.hostname or ''
    path = unquote(parsed.path).strip('/')
    if host.removeprefix('www.') == 'codeforces.com':
        match = re.fullmatch(r'problemset/problem/(\d+)/([A-Za-z]\d*)', path)
        if not match: match = re.fullmatch(r'contest/(\d+)/problem/([A-Za-z]\d*)', path)
        if match: return ('CODEFORCES', str(int(match[1])), match[2].upper())
        # Gym identity is deliberately separate from contest identity.
        match = re.fullmatch(r'gym/(\d+)/problem/([A-Za-z]\d*)', path)
        if match: return ('CODEFORCES_GYM', str(int(match[1])), match[2].upper())
    if host.removeprefix('www.') == 'codechef.com':
        match = re.search(r'(?:^|/)problems/([^/]+)$', path, re.IGNORECASE)
        if match: return ('CODECHEF', match[1].upper())
    if host.removeprefix('www.') == 'hackerearth.com':
        match = re.search(r'(?:^|/)problem/(?:algorithm|approximate|machine-learning)/([^/]+)$', path)
        if match: return ('HACKEREARTH', match[1].casefold())
    return None


def original_problem_keys(row):
    source = row['original_source']; name = row['original_name']
    result = []
    if source == 'CODEFORCES' and row['original_CF_contest_id'] is not None and row['original_CF_problem_index']:
        result.append(('CODEFORCES', str(row['original_CF_contest_id']), row['original_CF_problem_index'].upper()))
    if source == 'CODECHEF': result.append(('CODECHEF', name.upper()))
    if source == 'HACKEREARTH': result.append(('HACKEREARTH', name.casefold()))
    if source in {'ATCODER', 'AIZU'}:
        match = re.match(r'^(p\d{5})(?:\s|$)', name)
        if match: result.append(('CODENET', match[1]))
    return result


def prompt(problem, language):
    label = LANGUAGES[language][0]
    interface = 'Read standard input and write standard output as specified.'
    source_input = problem.input_file if problem.HasField('input_file') else ''
    source_output = problem.output_file if problem.HasField('output_file') else ''
    if source_input or source_output:
        interface = ('Use the original problem file interface: input file ' + repr(source_input or 'standard input')
                     + ', output file ' + repr(source_output or 'standard output') + '.')
    return ('Write a ' + label + ' solution to the following problem. ' + interface
            + ' Return only the complete code.\n\n' + problem.description)


def native_sequence(tokenizer, original_prompt, original_code):
    prefix = [1] + tokenizer.encode('User: ' + original_prompt + '\nAssistant:').ids
    response = tokenizer.encode(original_code).ids
    if any(value < 4 for value in prefix[1:] + response):
        return None, 'source_contains_reserved_native_token'
    if not response: return None, 'empty_original_program'
    ids = prefix + response + [2]
    if len(ids) > 4096: return None, 'complete_context_exceeds_4096'
    return (ids, [-100] * len(prefix) + response + [2]), None


def select_complete_solutions(problem, tokenizer, protected_program_hashes, maximum_per_language=3):
    """Return first complete fitting unique publisher positives per language.

    All source positives are still visited for protected-program exposure.
    Every original positive is accounted for as selected, redundant, oversized,
    reserved-token, invalid encoding, or beyond this declared presentation cap.
    No incorrect solution or private test output enters returned targets.
    """
    if type(maximum_per_language) is not int or maximum_per_language < 1:
        raise ValueError('Positive fixed per-problem/language presentation cap required')
    from collections import Counter
    counts = Counter(); selected = []; language_counts = Counter()
    selected_digests = set(); protected_matches = []
    prompts = {language: prompt(problem, language) for language in LANGUAGES}
    for position, solution in enumerate(problem.solutions):
        code = solution.solution; language = int(solution.language)
        counts['original_publisher_correct_solutions_visited'] += 1
        if not isinstance(code, str):
            counts['original_non_UTF8_solution_review'] += 1
            continue
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        protected_hash = normalized_program_hash(code)
        if protected_hash in protected_program_hashes:
            protected_matches.append(dict(original_solution_position=position, actual_program_sha256=code_hash,
                protected_sources=protected_program_hashes[protected_hash]))
        if language not in LANGUAGES:
            counts['unsupported_original_language_label'] += 1
            continue
        if (language, code_hash) in selected_digests:
            counts['duplicate_selected_original_program'] += 1
            continue
        if language_counts[language] >= maximum_per_language:
            counts['beyond_fixed_per_language_presentation_cap'] += 1
            continue
        if not code or '\0' in code:
            counts['empty_or_NUL_original_program_review'] += 1
            continue
        sequence, reason = native_sequence(tokenizer, prompts[language], code)
        if sequence is None:
            counts[reason] += 1
            continue
        ids, labels = sequence
        selected.append(dict(original_solution_position=position, language=LANGUAGES[language][1],
            original_language_number=language, original_program_sha256=code_hash,
            prompt=prompts[language], response=code, ids=ids, labels=labels))
        selected_digests.add((language, code_hash)); language_counts[language] += 1
        counts['selected_complete_original_programs'] += 1
    if sum(v for k, v in counts.items() if k != 'original_publisher_correct_solutions_visited') != len(problem.solutions):
        raise ValueError('Every original positive must have one declared presentation disposition')
    return selected, dict(counts), protected_matches
