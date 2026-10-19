"""Check learner-visible original HH feedback text for benchmark exposure."""
from datetime import datetime, timezone
import json
from pathlib import Path

from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.exposure_audit import StreamMatcher
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'reports/original-feedback-benchmark-exposure-v1.json'
    if output.exists(): raise ValueError('Preserve existing original-feedback exposure audit')
    patterns = {}; digests = {}
    for family, name, field in [
        ('knowledge', 'data/knowledge-eval-v1/questions.jsonl', 'question'),
        ('math', 'data/math-reasoning-v2/test.jsonl', 'question'),
        ('public_code', 'data/public-code-assessment-v1/questions.jsonl', 'specification')]:
        path = ROOT / name; digests[str(path)] = file_hash(path)
        for row in jsonl(path): patterns[family + ':' + str(row['id'])] = row[field]
    tokenizer = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    digests[str(tokenizer)] = file_hash(tokenizer)
    tok = Tokenizer.from_file(str(tokenizer)); config = ModelConfig.load(ROOT / 'configs/model.json')
    data = ROOT / 'data/human-preference-curriculum-v1'
    manifest = json.loads((data / 'manifest.json').read_text()); digests[str(data / 'manifest.json')] = file_hash(data / 'manifest.json')
    encoder = Encoder(tok, config, data, 448)
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__), ROOT / 'src/slm_lab/exposure_audit.py']
    runtime = capture(files); findings = {}
    path = data / 'preferences.train.jsonl'; digests[str(path)] = file_hash(path)
    if digests[str(path)] != manifest['files_sha256'][path.name]: raise ValueError('Original human preference records changed')
    for choice in ['chosen', 'rejected']:
        matcher = StreamMatcher(tok, patterns); count = 0; previous_truncated = encoder.truncated
        for row in jsonl(path):
            ids, _ = encoder.sequence({**row, 'response': row[choice]}, 'response')
            matcher.feed(ids, path.name + ':' + choice); count += 1
        findings[choice] = {'source_sha256': digests[str(path)], 'records': count,
                            'truncated_sequences': encoder.truncated - previous_truncated, 'matches': matcher.finish()}
    path = data / 'ppo.train.jsonl'; digests[str(path)] = file_hash(path)
    if digests[str(path)] != manifest['files_sha256'][path.name]: raise ValueError('Original PPO prompts changed')
    matcher = StreamMatcher(tok, patterns); count = 0
    for row in jsonl(path):
        prompt = encoder.tokens(encoder.prompt(row))[-(config.max_text_length - 256 - 1):]
        matcher.feed([encoder.bos] + prompt + [encoder.eos], path.name); count += 1
    findings['PPO_prompts'] = {'source_sha256': digests[str(path)], 'records': count, 'rollout_max_new_tokens': 256,
                              'matches': matcher.finish()}
    if capture(files) != runtime: raise ValueError('Loaded feedback scanner/encoder changed')
    for path, digest in digests.items():
        if file_hash(path) != digest: raise ValueError('Original benchmark/feedback input changed during scan')
    report = {'completed_at': datetime.now(timezone.utc).isoformat(), 'findings': findings,
              'exposed_case_ids': sorted({key for item in findings.values() for key in item['matches']}),
              'source_files_sha256': digests, 'implementation_runtime': runtime,
              'benchmark_files_modified': False, 'goal_achieved': False,
              'scope': 'Exact benchmark question/specification text in actual encoder-visible chosen/rejected examples and truncated PPO prompts; generated PPO rollouts, paraphrases and semantic overlap are not covered.'}
    atomic_json(output, report); print(json.dumps({'records': {key: item['records'] for key, item in findings.items()},
                 'exposed_case_ids': report['exposed_case_ids']}), flush=True)


if __name__ == '__main__': main()
