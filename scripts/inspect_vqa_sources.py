"""Inspect original real-photo VQA pairs, human agreement and per-image licenses."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

from slm_lab.clevr_programs import array_items
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import asset, atomic_json

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/vqa-source-cache-v1'


def agreed_answer(annotation):
    answers = annotation['answers']
    if len(answers) != 10 or sorted(item['answer_id'] for item in answers) != list(range(1, 11)):
        raise ValueError('Original question does not have ten distinct human answer records')
    counts = Counter(item['answer'] for item in answers)
    target = annotation['multiple_choice_answer']; votes = counts[target]
    if votes != max(counts.values()): raise ValueError('Original declared most frequent answer differs from votes')
    return target, votes


def is_pair(first, second):
    first_question = ' '.join(first['question']['question'].casefold().split())
    second_question = ' '.join(second['question']['question'].casefold().split())
    return (first_question == second_question and first['question']['image_id'] != second['question']['image_id']
            and first['answer'] != second['answer'])


def main():
    output = ROOT / 'reports/vqa-original-source-inspection-v1.json'
    if output.exists(): raise ValueError('Preserve existing inspection evidence; do not silently overwrite')
    sources = {}
    for name in ['coco-annotations-trainval2014.zip', 'vqa2-train-questions.zip', 'vqa2-valid-questions.zip',
                 'vqa2-train-answers.zip', 'vqa2-valid-answers.zip', 'download.html', 'terms.html', 'coco-terms.html']:
        receipt = json.loads((CACHE / (name + '.source.json')).read_text())
        if file_hash(CACHE / name) != receipt['sha256']: raise ValueError('Cached original source bytes changed: ' + name)
        sources[name] = receipt
    for split, part in [('train', 'Train'), ('valid', 'Val')]:
        name = 'vqa2-' + split + '-pairs.zip'
        _, receipt = asset(CACHE, name, 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Complementary_Pairs_' + part + '_mscoco.zip', 20_000_000)
        sources[name] = receipt
    split_stats = {}; inventories = {}; files = {}
    with zipfile.ZipFile(CACHE / 'coco-annotations-trainval2014.zip') as coco:
        for split, original_part in [('train', 'train2014'), ('valid', 'val2014')]:
            member = 'annotations/captions_' + original_part + '.json'
            # COCO puts licenses after its large images array. The CLEVR
            # reader's deliberately small header limit cannot locate that key.
            # Caption members are bounded66/32MB, so parse each complete member.
            with coco.open(member) as stream: captions = json.load(stream)
            images = {image['id']: image for image in captions['images']}
            inventories[split] = set(images)
            licenses = {license['id']: license for license in captions['licenses']}
            del captions
            if licenses[4]['url'] != 'http://creativecommons.org/licenses/by/2.0/':
                raise ValueError('Original COCO license identifier4 is not expected CC-BY2.0')
            image_counts = Counter(image['license'] for image in images.values())
            licensed_images = {key: value for key, value in images.items() if value['license'] == 4}
            with zipfile.ZipFile(CACHE / ('vqa2-' + split + '-questions.zip')) as archive:
                with archive.open('v2_OpenEnded_mscoco_' + original_part + '_questions.json') as stream: questions = json.load(stream)
            if (questions['data_subtype'] != original_part or questions['license']['url'] != 'http://creativecommons.org/licenses/by/4.0/'):
                raise ValueError('Original question split/license differs')
            questions_by_id = {question['question_id']: question for question in questions['questions']}
            if len(questions_by_id) != len(questions['questions']): raise ValueError('Repeated original question identity')
            question_license = questions['license']; del questions
            eligible = {}; exclusions = Counter(); seen = set(); answer_types = Counter(); agreement = Counter()
            with zipfile.ZipFile(CACHE / ('vqa2-' + split + '-answers.zip')) as archive:
                member = 'v2_mscoco_' + original_part + '_annotations.json'
                with archive.open(member) as stream:
                    prefix = stream.read(65536).decode('utf-8'); marker = re.search(r'"annotations"\s*:\s*\[', prefix)
                    if not marker: raise ValueError('Original annotation header absent from bounded prefix')
                    header = json.loads(prefix[:marker.start()].rstrip().rstrip(',') + '}')
                if header['data_subtype'] != original_part or header['license'] != question_license:
                    raise ValueError('Original answer split/license differs from questions')
                with archive.open(member) as stream:
                    for annotation in array_items(stream, 'annotations'):
                        qid = annotation['question_id']
                        if qid in seen or qid not in questions_by_id: raise ValueError('Repeated/missing original question/answer join')
                        seen.add(qid); question = questions_by_id[qid]
                        if question['image_id'] != annotation['image_id'] or question['image_id'] not in images:
                            raise ValueError('Original image/answer/question identities differ')
                        answer, votes = agreed_answer(annotation)
                        if annotation['image_id'] not in licensed_images:
                            exclusions['original image license outside CC-BY2.0 subset'] += 1; continue
                        agreement[str(votes)] += 1
                        if votes < 6: exclusions['less than6/10 exact human agreement'] += 1; continue
                        eligible[qid] = {'question': question, 'annotation': annotation, 'image': licensed_images[annotation['image_id']],
                                         'answer': answer, 'exact_human_votes': votes, 'annotation_license': question_license,
                                         'image_license': licenses[4], 'training_admitted': False}
                        answer_types[annotation['answer_type']] += 1
            if seen != set(questions_by_id): raise ValueError('Original questions and answer inventory differ')
            with zipfile.ZipFile(CACHE / ('vqa2-' + split + '-pairs.zip')) as archive:
                if len(archive.namelist()) != 1: raise ValueError('Original paired-question ZIP member inventory differs')
                with archive.open(archive.namelist()[0]) as stream: pairs = json.load(stream)
            retained = []; pair_exclusions = Counter(); pair_ids = set(); selected_images = set()
            for pair in pairs:
                if len(pair) != 2 or not all(type(qid) is int and qid in questions_by_id for qid in pair):
                    raise ValueError('Original paired-question identity invalid')
                identity = tuple(sorted(pair))
                if identity in pair_ids: pair_exclusions['duplicate original pair'] += 1; continue
                pair_ids.add(identity)
                if not all(qid in eligible for qid in pair): pair_exclusions['pair lacks two licensed high-agreement questions'] += 1; continue
                first, second = [eligible[qid] for qid in pair]
                if not is_pair(first, second): pair_exclusions['question/image/answer contrast differs'] += 1; continue
                selected_images.update([first['question']['image_id'], second['question']['image_id']])
                retained.append({'id': 'vqa2:' + original_part + ':pair:' + '-'.join(map(str, identity)), 'source_subset': original_part,
                                 'first': first, 'second': second, 'training_admitted': False})
            path = CACHE / ('candidate-pairs.' + split + '.jsonl')
            if path.exists(): raise ValueError('Candidate inspection rows already exist; preserve prior run')
            path.write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in retained))
            files[path.name] = file_hash(path)
            split_stats[split] = {'original_images': len(images), 'original_license_counts': dict(sorted(image_counts.items())),
                                 'original_CC_BY_2_images': len(licensed_images), 'original_questions': len(questions_by_id),
                                 'high_agreement_CC_BY_2_questions': len(eligible), 'answer_types': dict(answer_types),
                                 'CC_BY_2_exact_agreement_histogram': dict(agreement), 'question_exclusions': dict(exclusions),
                                 'original_pairs': len(pairs), 'eligible_complementary_pairs': len(retained),
                                 'eligible_pair_images': len(selected_images), 'pair_exclusions': dict(pair_exclusions)}
            print('Original real-photo VQA metadata inspected', split, split_stats[split], flush=True)
    if inventories['train'] & inventories['valid']: raise ValueError('Original COCO train/val image identities overlap')
    report = {'inspected_at': datetime.now(timezone.utc).isoformat(), 'publisher': 'https://visualqa.org/download.html',
              'sources': sources, 'splits': split_stats, 'candidate_files_sha256': files,
              'inspection_source_sha256': file_hash(__file__), 'training_admitted': False, 'images_downloaded': 0,
              'goal_achieved': False, 'external_model_teacher_used': False,
              'scope': 'Original human-labelled real-photo source eligibility and paired-question inspection; no training or capability claim.',
              'limitations': ['CC-BY2.0 metadata subset, not the full VQA benchmark or independent verification of current photo rights.',
                              'Image pixels, existing held-out aliases/near-duplicates and creator attribution still require collection/audit before training.',
                              'Original answers contain human disagreement;6/10 exact agreement is an admission heuristic, not a label correctness guarantee.',
                              'Original COCO endpoint has TLS hostname mismatch; metadata fetched via verified HTTPS address of the same public S3 bucket.']}
    atomic_json(output, report)


if __name__ == '__main__': main()
