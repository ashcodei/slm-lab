"""Complete text-only problem/program encoding for the new coding curriculum."""
from .training import Encoder


class CompleteContextDoesNotFit(ValueError):
    def __init__(self, prompt_tokens, response_tokens, limit):
        self.prompt_tokens = prompt_tokens
        self.response_tokens = response_tokens
        self.limit = limit
        super().__init__(f'Complete original problem/program requires {prompt_tokens + response_tokens + 2} tokens; limit {limit}')


class CompleteCodeEncoder(Encoder):
    """Use the complete text window without clipping either side of an SFT row."""
    def sequence(self, row, response_field=None):
        if response_field is None:
            return super().sequence(row, response_field)
        if response_field != 'response' or row.get('image') or 'tokens' in row:
            raise ValueError('Complete code encoding requires a plain text response row')
        prompt = self.tokens(self.prompt(row))
        response = self.tokens(row['response'])
        if not response:
            raise ValueError('A complete code target must be nonempty')
        if len(prompt) + len(response) + 2 > self.config.max_text_length:
            raise CompleteContextDoesNotFit(len(prompt), len(response), self.config.max_text_length)
        ids = [self.bos] + prompt + response + [self.eos]
        labels = [-100] * (len(prompt) + 1) + response + [self.eos]
        return ids, labels


def problem_program_row(problem, metadata, code):
    """Format original text/code without adding solutions or preference labels."""
    if metadata['status'] != 'Accepted' or metadata['problem_id'] != problem['problem_id']:
        raise ValueError('Original accepted problem/source identity differs')
    if metadata['language'] == 'Text':
        raise ValueError('Output-only Text submissions are not program targets')
    if not code.strip() or '\0' in code or not problem['text'].strip():
        raise ValueError('Complete nonempty original problem/program text required')
    if problem['requires_media_review']:
        raise ValueError('Original problem needs unresolved media review')
    if any('Unclosed' in issue['reason'] for issue in problem['sample_parse_issues']):
        raise ValueError('Original problem markup needs review')
    version = metadata['original_language'] or metadata['language']
    prompt = (f"Write a complete {metadata['language']} program.\n"
              f"Target language/runtime: {version}.\n"
              "Use standard input and output as the problem specifies. Return only the complete program.\n\n"
              + problem['text'])
    return {'id': 'codenet-program:' + metadata['submission_id'],
        'source_problem_id': metadata['problem_id'], 'group_id': None, 'split': None,
        'prompt': prompt, 'response': code, 'language': metadata['language'],
        'original_language': metadata['original_language'],
        'source': 'IBM Project CodeNet original submission', 'license': 'CDLA-Permissive-2.0',
        'original_judge_status': metadata['status'],
        'annotation_origin': 'Original public problem and human program, with static task framing; no generated solution or human preference label.'}
