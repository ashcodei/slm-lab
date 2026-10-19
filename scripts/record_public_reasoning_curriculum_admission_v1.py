"""Record complete source curriculum admission while preserving final scope."""
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.function_repair_GPU_contract_v1 import ADMISSION,QUEUE,handoff_admitted

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-reasoning-curriculum-admission-v1-review'
DATA=ROOT/'data/public-reasoning-curriculum-v1'
PRODUCER=ROOT/'reports/public-reasoning-curriculum-CPU-v1'
READER=ROOT/'reports/public-reasoning-curriculum-v1-readback-admission'


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    pending.replace(path)


def live(owner):
    row=(Path('/proc')/str(owner['pid'])/'stat').read_text().split(') ',1)[1].split()
    if row[19]!=owner['ticks'] or row[0] in {'Z','X'}:raise ValueError('Exact original process birth must remain live')
    return dict(owner=owner,state=row[0])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists():raise ValueError('Fresh low-priority CPU admission review required')
    reader=read(READER/'complete.json');producer=read(PRODUCER/'complete.json');admission=read(DATA/'admission.json')
    if (not reader.get('complete') or not reader['admitted_to_training'] or reader['new_neural_updates']!=0
            or reader['complete_original_draws_regenerated']!=3072 or reader['complete_protected_input_stems_rechecked']!=31349
            or reader['actual_packed_sequences_rebuilt']!=2857 or reader['actual_counts']['train']['examples']!=2568
            or reader['actual_counts']['validation']['examples']!=289 or reader['admission_file_sha256']!=sha(DATA/'admission.json')
            or reader['dataset_manifest_sha256']!=sha(DATA/'manifest.json') or not reader['actual_admitted_training_access_verified']
            or not reader['actual_unadmitted_training_rejection_observed'] or not reader['actual_both_held_training_rejections_observed']):
        raise ValueError('Complete actual public-source/native/overlap/access admission evidence required')
    audit=read(ROOT/'reports/completion-audit.json');previous=Path(audit['latest_full_scope_review']);review=copy.deepcopy(read(previous))
    gpu=read(ROOT/QUEUE);handoff=read(ROOT/ADMISSION)
    if not handoff_admitted(ROOT,gpu['owner']):raise ValueError('All ten original full-scope successor owners must remain admitted')
    maps=[read(ROOT/'reports/public-pretraining-recovery-v1/plan.json')['source_files_sha256'],
          read(ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json')['source_files_sha256'],
          read(ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json')['source_files_sha256'],
          read(ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json')['source_files_sha256'],gpu['source_files_sha256']]
    maps += [read(ROOT/item['queue'])['source_files_sha256'] for item in handoff['new_controllers']]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in mapping for mapping in maps):raise ValueError('Never edit a live source-bound index/document')
    files=dict(reader['source_files_sha256'])
    for path in [Path(__file__),previous,READER/'complete.json',DATA/'admission.json',
        ROOT/'reports/reasoning-curriculum-roles-v1-tests.log',ROOT/'reports/public-reasoning-protected-text-v1-tests.log']:
        files[str(path)]=sha(path)
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Every source/producer/artifact must remain exact')
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
    rows=list(map(json.loads,prefix.splitlines()))
    if [row['step'] for row in rows]!=list(range(1,len(rows)+1)):raise ValueError('Actual complete contiguous main prefix required')
    current=dict(owner=dict(pid=1241,ticks='49351203'),complete_actual_updates=len(rows),planned_updates=4447,
        response_EOS_targets=sum(int(row['response_EOS_targets']) for row in rows),complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    active=[live(owner) for owner in [current['owner'],dict(pid=18545,ticks='50128787'),dict(pid=25240,ticks='51635952'),
        gpu['owner'],read(ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json')['owner']]]
    if any(row['state'] in {'T','t'} for row in active):raise ValueError('Original active neural work and optional research must remain running')
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    proof=dict(complete=True,at=at,source_files_sha256=files,
        previous_goal_turn_classification='progress: actual pinned twelve-task public source qualification and full fresh-process source/native/reward readback',
        current_goal_turn_classification='progress: actual full protected-input screen, input-only equivalent-group split, complete native packs and independent training admission',
        actual_admission=reader,actual_source_draw_dispositions=producer['actual_original_dispositions'],actual_task_counts=producer['actual_per_task'],
        actual_original_rows_with_declared_text_overlap=producer['actual_original_draws_with_text_overlap'],
        actual_train_validation_group_intersection=0,actual_new_grouping_and_protected_population_tests_passed=10,
        original_main=current,actual_live_original_processes=active,
        original_CPU_comparison=read(ROOT/'reports/full-function-repair-comparison-CPU-v1/progress.json'),
        original_CPU_assessment=read(ROOT/'reports/public-pretraining-interim-CPU-v2/progress.json'),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        all_original17families_human_RL_and_nine_endpoint_benchmarks_preserved=True,
        all_ten_original_full_scope_successor_owners_live=True,new_neural_updates_from_new_curriculum=0,
        model_has_consumed_new_curriculum=False,candidate_promoted=False,goal_achieved=False)
    atomic(OUT/'complete.json',proof)
    lines=['# Public reasoning curriculum admission','',
        'The prior turn qualified 3,072 actual questions and strict rewards from twelve pinned, unchanged public generators. '
        'This turn completes input overlap control, groups declared equivalent inputs before splitting, and independently admits the resulting whole native training partition.', '',
        '| Partition | Whole examples | Context tokens | Response/EOS targets |','|---|---:|---:|---:|']
    for split in ['train','validation']:
        value=reader['actual_counts'][split]
        lines.append(f'| {split} | {value["examples"]:,} | {value["context_tokens"]:,} | {value["target_tokens"]:,} |')
    lines += ['', 'All **3,072** original draws have recorded dispositions. There are **2,857** declared input groups and **215** duplicate/equivalent draws. '
        'Groups include reordered GCD/LCM inputs, signed additive terms, multiplicative factors, string-pattern/letter renamings and specified grid symmetries. '
        'Every one of the twelve tasks retains both training and reserved groups. The fixed role hash never reads public answers. '
        'The actual train/reserved group intersection is zero.', '',
        'The complete candidate prompt and answer undergo normalized literal and rare informative five-word-shingle checks against **31,349 original input stems**. '
        'A separate reverse literal scan looks for whole original candidate questions inside protected inputs. '
        'The protected registry includes all 16,961 QA questions plus 600 MBPP problems, 5,000 APPS test prompts, 3,088 AI2D questions, '
        '4,642 held human preference prompts, 677 dialogue prompts, all 203 broad validation texts, all 151 local pulse source inputs, '
        '15 original source-function validation prompts and 12 local coding specifications. No protected gold label chooses a split. '
        'No source row matches these declared checks. Any match would exclude the entire input group, including equivalent siblings.', '',
        'The independent reader actually regenerates every original public draw, independently recomputes every answer, reruns every overlap check and role disposition, '
        'and rebuilds all **2,857** packed sequences and target masks. Whole original questions, canonical public answers and true EOS are retained without truncation. '
        'The actual loader rejects unadmitted training access, rejects both held-role optimization requests, then accepts exactly the admitted **2,568** training examples. '
        'Seven input-group tests and three protected-population/overlap controls pass. The precondition failure and initial Unicode-line/OCR-population test failures are preserved; '
        'the final complete population retains embedded Unicode paragraph separators and all 48 OCR source rows.', '',
        '**Admission is a data result, not a model result. No model has consumed the new curriculum and it has produced zero neural updates.** '
        'The 289 reserved examples are reserved from this new curriculum only. These lexical checks and declared transformations do not certify arbitrary semantic equivalence '
        'or independence from all historical training bytes. Model evaluation must retain that exposure limitation.', '',
        f'The original main remains live at **{len(rows):,}/4,447** updates and **{current["response_EOS_targets"]:,}** actual response/EOS targets. '
        'All four L40s remain assigned to that run. Existing CPU training/assessment, the queued preference comparison, all ten downstream owners, '
        '17 source families and full human/DPO/execution-RL/nine-endpoint evaluations remain intact. '
        '**Latest complete local functional correctness remains 0/48; no new model-quality improvement is established. All ten final requirements remain incomplete.**','']
    (OUT/'metrics.md').write_text('\n'.join(lines))
    review['requirements']['public_data']['evidence']+=(
        ' The full3072draw public reasoning population is now screened against31349original protected input stems in both literal directions plus rare-shingle copies. '
        'Declared input-equivalence groups yield2857representatives:2568train/289reserved,215duplicates,0declared text matches and0train/reservedgroupintersection. '
        'A fresh actual readback regenerates every original draw, reruns the complete overlap screen, and rebuilds all2857whole native masks. '
        'The actual loader rejects unadmitted and held optimization access, then accepts all2568admitted train examples/19080responseEOS targets. '
        'All12tasks retain nonempty train and reserved groups. No model consumed them; historical or arbitrary semantic independence is not certified.')
    if len(review['requirements'])!=10 or any(v['status']!='incomplete_for_requested_final_model' for v in review['requirements'].values()):
        raise ValueError('A data admission proves none of the requested final model capability requirements')
    review.update(at=at,previous_full_scope_review=str(previous),previous_goal_turn_classification=proof['previous_goal_turn_classification'],
        current_goal_turn_classification=proof['current_goal_turn_classification'],latest_actual_reasoning_curriculum_admission=proof,
        source_files_sha256={**files,str(OUT/'complete.json'):sha(OUT/'complete.json')},goal_achieved=False)
    atomic(OUT/'review.json',review)
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('All source bindings must stay exact before index update')
    audit.update(at=at,requirements=review['requirements'],latest_full_scope_review=str(OUT/'review.json'),latest_actual_main_runtime=current,
        latest_actual_reasoning_curriculum_admission=str(OUT/'complete.json'),goal_achieved=False)
    atomic(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f'Public reasoning curriculum admitted, **{at}**: every one of3,072original draws is screened against **31,349 protected input stems** '
        'and assigned by input-only equivalence group before splitting. There are **2,568 admitted training examples**, **289 reserved examples**, '
        '**215 duplicate/equivalent draws**, no declared text matches and zero train/reserved group intersection. '
        'A fresh reader regenerates every source draw and rebuilds all2,857whole native masks;10new focused controls pass. '
        '**No model has consumed this curriculum; no model gain or historical independence is claimed.** '
        f'The main continues at **{len(rows)}/4,447** with all original source/human/RL/nine-endpoint benchmark scopes. '
        '**Latest complete functional correctness remains0/48; all ten final requirements remain incomplete.** '
        '[Admission and limits](reports/public-reasoning-curriculum-admission-v1-review/metrics.md), '
        '[full review](reports/public-reasoning-curriculum-admission-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),admitted_train_examples=2568,main_actual_updates=len(rows),goal_achieved=False)),flush=True)


if __name__=='__main__':main()
