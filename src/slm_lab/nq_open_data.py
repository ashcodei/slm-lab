"""Original publisher human question/answer supervision for our own language model."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata

from tokenizers import Tokenizer

from .code_tasks import TASKS
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture
from .model import ModelConfig
from .training import Encoder

ROOT = Path(__file__).resolve().parents[2]
REVISION = 'fb26a3073b1fe636c97302890a27b491d6530130'
REPOSITORY = 'https://github.com/google-research-datasets/natural-questions'


def canonical(question):
    return ' '.join(unicodedata.normalize('NFKC', question).casefold().split())


def example(original, split, line):
    if (set(original) != {'question', 'answer'} or not isinstance(original['question'], str)
            or not original['question'].strip() or not isinstance(original['answer'], list) or not original['answer']
            or any(not isinstance(value, str) or not value.strip() for value in original['answer'])):
        raise ValueError('Original publisher NQ-open human question/answer schema differs')
    if split not in ['train', 'dev']: raise ValueError('Use original NQ-open train/dev splits')
    question = original['question']; aliases = list(original['answer'])
    digest = hashlib.sha256(canonical(question).encode()).hexdigest()
    return {'id': 'nq-open:' + split + ':' + str(line) + ':' + digest,
            'group_id': 'nq-open-question:' + digest, 'source_line': line, 'source_split': split,
            'split': 'train' if split == 'train' else 'valid', 'task': 'nq_open_question',
            'domain': 'open_domain_knowledge', 'language': 'English',
            'prompt': question + '\nAnswer briefly.', 'response': aliases[0], 'reference_answers': aliases,
            'source_original_question': original, 'feedback_source': 'human',
            'answer_origin': 'First original publisher reference string, unchanged; all original aliases retained. No model-generated answer.',
            'source_url': REPOSITORY + '/blob/' + REVISION + '/nq_open/NQ-open.' + split + '.jsonl#L' + str(line),
            'source_revision': REVISION, 'license': 'CC BY-SA 3.0',
            'license_url': 'https://creativecommons.org/licenses/by-sa/3.0/',
            'attribution': 'Natural Questions / NQ-Open contributors; Kwiatkowski et al. (2019), Lee et al. (2019).',
            'modifications': 'Question wrapped as a brief-answer instruction; all original reference strings preserved.'}


def prepare(output):
    output = Path(output)
    if output.exists(): raise ValueError('Preserve existing NQ instruction preparation; inspect before resuming')
    source = ROOT / 'data/nq-open-original-source-v1'; source_manifest_path = source / 'manifest.json'
    source_manifest = json.loads(source_manifest_path.read_text()); source_manifest_sha = file_hash(source_manifest_path)
    if (source_manifest['revision'] != REVISION or source_manifest['data_license'] != 'CC BY-SA 3.0'
            or source_manifest['implementation_sha256'] != file_hash(ROOT / 'scripts/fetch_nq_open_source.py')):
        raise ValueError('Original publisher source/revision/data-license receipt differs')
    for name, item in source_manifest['sources'].items():
        raw = (source / name).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if file_hash(source / name) != item['sha256'] or blob != item['publisher_git_blob_sha1']:
            raise ValueError('Original publisher NQ source bytes/blob identity changed')
    rows = {split: list(jsonl(source / ('nq_open/NQ-open.' + split + '.jsonl'))) for split in ['train', 'dev']}
    if len(rows['train']) != 87925 or len(rows['dev']) != 3610: raise ValueError('Original full source split counts differ')
    protected = {canonical(row['question']) for row in rows['dev']}
    reserved_paths = [ROOT / name for name in ['data/knowledge-eval-v1/questions.jsonl',
                       'data/math-reasoning-v2/test.jsonl', 'data/public-code-assessment-v1/questions.jsonl']]
    reserved_sha = {str(path): file_hash(path) for path in reserved_paths}
    for path in reserved_paths:
        for row in jsonl(path): protected.add(canonical(row.get('question', row.get('specification', ''))))
    protected.update(canonical(task[2]) for task in TASKS); protected.discard('')
    tokenizer_path = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    config_path = ROOT / 'configs/model.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path)); config = ModelConfig.load(config_path)
    files = [Path(__file__), ROOT / 'scripts/fetch_nq_open_source.py'] + [ROOT / 'src/slm_lab' / (name + '.py') for name in
             ['training', 'model', 'corpus', 'diverse_images', 'data', 'code_tasks', 'evaluation_provenance']]
    runtime = capture(files); tokenizer_sha = file_hash(tokenizer_path); config_sha = file_hash(config_path)
    output.mkdir(parents=True); encoder = Encoder(tokenizer, config, output, 448)
    statistics = {}; groups = {}; data_sha = {}
    for split in ['train', 'dev']:
        stats = Counter(); accepted = []; seen = set()
        for number, original in enumerate(rows[split], 1):
            row = example(original, split, number); stats['source_rows'] += 1
            if row['group_id'] in seen: raise ValueError('Repeated canonical original source question')
            seen.add(row['group_id'])
            if split == 'train' and canonical(original['question']) in protected:
                stats['excluded_exact_reserved_question'] += 1; continue
            truncated = encoder.truncated; _, labels = encoder.sequence(row, 'response')
            if encoder.truncated != truncated:
                stats['excluded_complete_question_answer_context_budget'] += 1; continue
            row['target_tokens'] = sum(value != -100 for value in labels[1:])
            stats['examples'] += 1; stats['target_token_exposures_per_pass'] += row['target_tokens']; accepted.append(row)
        name = 'train.jsonl' if split == 'train' else 'valid.jsonl'
        with (output / name).open('w') as writer:
            for row in accepted: writer.write(json.dumps(row, ensure_ascii=False) + '\n')
        data_sha[name] = file_hash(output / name); statistics[split] = dict(stats)
        groups[split] = {row['group_id'] for row in accepted}
    if groups['train'] & groups['dev']: raise ValueError('Human question groups cross training/validation splits')
    if statistics['dev']['examples'] != 3610: raise ValueError('Full original NQ-open development population must be retained')
    if (capture(files) != runtime or file_hash(source_manifest_path) != source_manifest_sha
            or file_hash(tokenizer_path) != tokenizer_sha or file_hash(config_path) != config_sha
            or any(file_hash(path) != digest for path, digest in reserved_sha.items())):
        raise ValueError('NQ preparation source/tokenizer/config/protected questions changed')
    manifest = {'prepared_at': datetime.now(timezone.utc).isoformat(), 'source_manifest_sha256': source_manifest_sha,
                'source_revision': REVISION, 'source_provenance': runtime, 'tokenizer_sha256': tokenizer_sha,
                'model_config_sha256': config_sha, 'reserved_question_sources_sha256': reserved_sha,
                'files_sha256': data_sha, 'splits': statistics, 'training_validation_group_overlap': 0,
                'all_original_reference_aliases_preserved': True, 'original_answer_texts_unchanged': True,
                'reference_answers_supplied_in_prompt': False, 'external_teacher_or_weights_used': False,
                'training_started': False, 'training_admitted': False, 'goal_achieved': False,
                'scope': 'Prepared original human open-domain brief-answer instruction rows; official development questions and existing exact reserved questions excluded from new training.',
                'limitations': ['Prior training/existing queued source exposure to these new development questions remains unaudited.',
                                'Original open-domain files omit Wikipedia page identity; article/paraphrase grouping and semantic overlap are not established.',
                                'First source answer is a training target; aliases and annotation disagreements remain in source metadata.',
                                'English fact QA is not expert coverage of all requested domains or independent final capability proof.']}
    atomic_json(output / 'manifest.json', manifest); return manifest


if __name__ == '__main__':
    report = prepare(ROOT / 'data/nq-human-questions-v1')
    print(json.dumps({key: value for key, value in report.items() if key != 'source_provenance'}), flush=True)
