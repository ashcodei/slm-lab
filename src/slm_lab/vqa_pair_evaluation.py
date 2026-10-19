"""Original real-photo paired questions with actual/blank/counterpart controls."""
import argparse
from collections import defaultdict, deque
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
from tokenizers import Tokenizer
from .clevr_evaluation import compare_reports as compare_questions
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl, pixel_hash
from .evaluation_provenance import capture, pixels_sha256, seal
from .object_evaluation import summarize
from .ocr_evaluation import normalized
from .training import Encoder, load_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def text(value):
    return normalized(value).casefold()


def human_agreement(prediction, answers):
    if len(answers) != 10 or sorted(row['answer_id'] for row in answers) != list(range(1, 11)):
        raise ValueError('Expected ten distinct original human answers')
    values = [text(row['answer']) for row in answers]; value = text(prediction)
    matches = sum(answer == value for answer in values)
    return sum(min(1., (matches - int(answer == value)) / 3) for answer in values) / 10


def validate_pairs(pairs, rows):
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows) or len({pair['id'] for pair in pairs}) != len(pairs):
        raise ValueError('Repeated real-photo question/pair identity')
    for pair in pairs:
        ids = pair['question_ids']
        if len(ids) != 2 or ids[0] == ids[1] or any(key not in by_id for key in ids):
            raise ValueError('Incomplete original visual question pair')
        first, second = [by_id[key] for key in ids]
        prompt = pair['canonical_question'] + '\nAnswer briefly.'
        if (first['prompt'] != prompt or second['prompt'] != prompt or first['group_id'] == second['group_id']
                or text(first['response']) == text(second['response'])):
            raise ValueError('Pair must keep identical model question and change image/target')
        for row in [first, second]:
            answers = row['original_human_annotation']['answers']
            human_agreement(row['response'], answers)
            if sum(answer['answer'] == row['response'] for answer in answers) < 6:
                raise ValueError('Original paired target lacks declared human consensus')
    return by_id


def select(pairs, rows, count=128):
    by_id = validate_pairs(pairs, rows); strata = defaultdict(deque)
    for pair in sorted(pairs, key=lambda row: hashlib.sha256(row['id'].encode()).digest()):
        first, second = [by_id[key] for key in pair['question_ids']]
        stratum = tuple(sorted([first['original_human_annotation']['answer_type'], second['original_human_annotation']['answer_type']]))
        strata[stratum].append(pair)
    chosen = []; used = set()
    while strata and len(chosen) < count:
        for kind in sorted(list(strata)):
            queue = strata[kind]
            while queue:
                pair = queue.popleft(); groups = {by_id[key]['group_id'] for key in pair['question_ids']}
                if not groups & used:
                    chosen.append(pair); used.update(groups); break
            if not queue: del strata[kind]
            if len(chosen) >= count: break
    return chosen, by_id


def paired_accuracy(results):
    grouped = defaultdict(list)
    for row in results: grouped[row['pair_id']].append(row)
    if not grouped or any(len(rows) != 2 or rows[0]['id'] == rows[1]['id'] for rows in grouped.values()):
        raise ValueError('Paired evaluation evidence is empty, incomplete or repeated')
    return {'pairs': len(grouped), 'both_answers_correct': sum(all(row['proper_exact'] for row in rows) for rows in grouped.values()),
            'both_answers_exact_accuracy': sum(all(row['proper_exact'] for row in rows) for rows in grouped.values()) / len(grouped)}


def compare_reports(before, after):
    for key in ['pairs_sha256', 'human_score_protocol']:
        if before[key] != after[key]: raise ValueError('Unmatched original real-photo paired assessment: ' + key)
    guard = compare_questions(before, after)
    if [row['pair_id'] for row in before['results']] != [row['pair_id'] for row in after['results']]:
        raise ValueError('Unmatched original pair memberships')
    first = paired_accuracy(before['results']); second = paired_accuracy(after['results'])
    if before['paired'] != first or after['paired'] != second: raise ValueError('Reported pair aggregate differs from actual cases')
    guard['passed'] = guard['passed'] and second['both_answers_exact_accuracy'] >= first['both_answers_exact_accuracy']
    guard.update(before_pairs=first, after_pairs=second, scope='Matched reused real-photo subset; not full VQA or independent final high-end assessment.')
    return guard


@torch.no_grad()
def evaluate(checkpoint, data, output, device='cuda:2', max_pairs=128, max_new_tokens=16):
    files = [ROOT / path for path in capture()['files_sha256']] + [Path(__file__), ROOT/'src/slm_lab/clevr_evaluation.py']
    provenance = capture(files); data = Path(data)
    manifest_digest = file_hash(data/'manifest.json')
    manifest = json.loads((data/'manifest.json').read_text()); data_path = data/'valid.jsonl'; pair_path = data/'pairs.valid.jsonl'
    digests = {path.name:file_hash(path) for path in [data_path,pair_path]}
    for path in [data_path,pair_path]:
        if digests[path.name] != manifest['files_sha256'][path.name]: raise ValueError('Original finalized validation questions/pairs changed')
    rows = list(jsonl(data_path)); pairs, by_id = select(list(jsonl(pair_path)), rows, max_pairs)
    if not pairs: raise ValueError('No complete independent-photo pairs available')
    model, meta = load_checkpoint(checkpoint, device, training_state=False); model.eval(); torch.set_num_threads(1)
    tokenizer_path = Path(checkpoint).parent/'tokenizer.json'
    if file_hash(tokenizer_path) != manifest['tokenizer_sha256'] or meta['tokenizer_sha256'] != manifest['tokenizer_sha256']:
        raise ValueError('Own model/paired-data tokenizer differs')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer_path)),model.config,data,448)
    amp = lambda: torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    results = []
    for number,pair in enumerate(pairs):
        samples = [by_id[key] for key in pair['question_ids']]
        if encoder.prompt(samples[0]) != encoder.prompt(samples[1]): raise ValueError('Actual paired model prompts differ')
        for position,row in enumerate(samples):
            other = samples[1-position]
            for sample in [row,other]:
                path = (data/sample['image']).resolve()
                if not path.is_relative_to(data.resolve()) or file_hash(path) != sample['stored_image_sha256']:
                    raise ValueError('Stored original pair image/path changed')
                with Image.open(path) as image:
                    if pixel_hash(image,448) != sample['model_pixel_sha256_448']: raise ValueError('Original paired input pixels changed')
            prompt = encoder.prompt(row); tokens = [encoder.bos]+encoder.tokens(prompt)
            if len(tokens)+max_new_tokens > model.config.max_text_length: raise ValueError('Full paired question exceeds context')
            ids = torch.tensor([tokens],device=device); pixels = encoder.pixels(row)[None].to(device); wrong = encoder.pixels(other)[None].to(device)
            batch = encoder.batch([row],'response',device); answers = row['original_human_annotation']['answers']
            result = {'id':row['id'],'pair_id':pair['id'],'group_id':row['group_id'],'expected':row['response'],
                      'task':row['original_human_annotation']['answer_type'],'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                      'input_pixels_sha256':pixels_sha256(pixels),'wrong_pixels_sha256':pixels_sha256(wrong),'wrong_image_id':other['id'],
                      'target_tokens':int((batch['labels'][:,1:]!=-100).sum())}
            for name,image in [('proper',pixels),('blank',torch.zeros_like(pixels)),('wrong',wrong)]:
                with amp():
                    generated = model.generate(ids,image,max_new_tokens=max_new_tokens,eos_token_id=encoder.eos)
                    nll = float(model(**{**batch,'pixel_values':image})['loss'])
                raw = encoder.tokenizer.decode(generated[0,len(tokens):].tolist())
                result[name+'_generated']=raw; result[name+'_exact']=text(raw)==text(row['response']);result[name+'_nll']=nll
                result[name+'_human_agreement']=human_agreement(raw,answers)
            results.append(result)
        if results[-1]['blank_generated'] != results[-2]['blank_generated']:
            raise ValueError('Identical blank image/question yielded different greedy outputs; pair protocol is invalid')
        if (number+1)%16==0:print('Original real-photo pairs evaluated',number+1,flush=True)
    if encoder.truncated: raise ValueError('Original paired visual answer targets truncated')
    report = {'checkpoint_sha256':file_hash(checkpoint),'data_sha256':digests[data_path.name],'pairs_sha256':digests[pair_path.name],
              'manifest_sha256':manifest_digest,'image_size':448,'max_examples':max_pairs*2,'max_new_tokens':max_new_tokens,
              'overall':summarize(results),'paired':paired_accuracy(results),'results':results,'goal_achieved':False,
              'mean_human_agreement':{name:sum(row[name+'_human_agreement'] for row in results)/len(results) for name in ['proper','blank','wrong']},
              'protocol':'Answer-type round-robin, unique photos across pairs; identical complete canonical model question, distinct original consensus targets; greedy448px/16tokens; actual/blank/paired counterpart controls.',
              'human_score_protocol':'Mean leave-one-annotator-out min(matches/3,1), ten original answers, with our NFKC/whitespace/case normalization only; not full official punctuation/number/contraction normalization.',
              'limitations':['Filtered reused original real-photo subset, not full standardized VQA or independent final high-end evidence.',
                             'Exact majority answer and both-correct pairs are primary; human agreement scores omit some official text normalization.',
                             'Wrong-image matches retain the original target; paired proper evaluations separately score the counterpart against its own human answers.']}
    seal(report,provenance,files=files,data=data_path,digest=report['data_sha256'],device=device,tokenizer_sha256=meta['tokenizer_sha256'])
    if file_hash(pair_path) != report['pairs_sha256']: raise ValueError('Original pair selection changed during assessment')
    if file_hash(data/'manifest.json') != manifest_digest: raise ValueError('Original pair manifest changed during assessment')
    atomic_json(Path(output),report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--data',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:2');args=parser.parse_args();evaluate(args.checkpoint,args.data,args.output,args.device)


if __name__=='__main__':main()
