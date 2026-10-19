"""Record exact tested causal/framing invariants without a model-gain claim."""
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import tokenizers
import torch

from slm_lab.function_repair_GPU_contract_v1 import ADMISSION,QUEUE,handoff_admitted
from slm_lab.local_source_closure_v2 import python_sources

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/active-causal-and-framing-v1-review'
TESTS=ROOT/'reports/active-causal-and-tokenizer-invariants-v1-tests.log'


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    pending.replace(path)


def live(owner):
    fields=(Path('/proc')/str(owner['pid'])/'stat').read_text().split(') ',1)[1].split()
    if fields[19]!=owner['ticks'] or fields[0] in {'Z','X','T','t'}:raise ValueError('Every active original/research process birth must remain live and running')
    return dict(owner=owner,state=fields[0])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists():raise ValueError('Fresh low-priority CPU causal/framing review required')
    if '25 passed' not in TESTS.read_text():raise ValueError('All exact source-path causal, gradient and framing checks must pass')
    audit=read(ROOT/'reports/completion-audit.json');previous=Path(audit['latest_full_scope_review']);review=copy.deepcopy(read(previous))
    runtime=read(ROOT/'reports/public-pretraining-recovery-v1/plan.json');warmup=read(ROOT/'reports/public-reasoning-warmup-CPU-v1-queue.json')
    gpu=read(ROOT/QUEUE);handoff=read(ROOT/ADMISSION)
    if not handoff_admitted(ROOT,gpu['owner']):raise ValueError('All ten original complete-scope successors must remain admitted')
    maps=[runtime['source_files_sha256'],warmup['source_files_sha256'],gpu['source_files_sha256'],
        read(ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json')['source_files_sha256'],
        read(ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json')['source_files_sha256'],
        read(ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json')['source_files_sha256']]
    maps += [read(ROOT/item['queue'])['source_files_sha256'] for item in handoff['new_controllers']]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in mapping for mapping in maps):raise ValueError('Never edit a live source-bound index/document')
    paths=[Path(__file__),TESTS,previous,ROOT/'reports/public-pretraining-recovery-v1/plan.json',
        ROOT/'reports/public-reasoning-warmup-CPU-v1-queue.json',ROOT/'data/public-reasoning-curriculum-v1/tokenizer.json',
        ROOT/'checkpoints/public-pretraining-v1/initial/tokenizer.json',
        *python_sources(ROOT,[ROOT/'tests/test_active_causal_and_tokenizer_invariants_v1.py'])]
    files={str(path):sha(path) for path in paths}
    core=['model.py','own_dense_growth_v1.py','own_code_delta_v1.py','own_dense_actor_v1.py']
    actual_core={}
    for name in core:
        path=str(ROOT/'src/slm_lab'/name);digest=sha(path)
        applicable=[mapping[path] for mapping in maps if path in mapping]
        if not applicable or any(value!=digest for value in applicable):raise ValueError('Tested runtime code must exactly match every applicable original source binding')
        actual_core[path]=digest
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
    rows=list(map(json.loads,prefix.splitlines()))
    if [row['step'] for row in rows]!=list(range(1,len(rows)+1)):raise ValueError('Actual complete contiguous neural prefix required')
    current=dict(owner=dict(pid=1241,ticks='49351203'),complete_actual_updates=len(rows),planned_updates=4447,
        response_EOS_targets=sum(int(row['response_EOS_targets']) for row in rows),complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    owners=[current['owner'],dict(pid=18545,ticks='50128787'),dict(pid=25240,ticks='51635952'),
        dict(pid=28339,ticks='51862171'),warmup['owner'],gpu['owner']]
    actual=[live(owner) for owner in owners]
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    proof=dict(complete=True,at=at,source_files_sha256=files,actual_core_bytes_match_live_source_bindings=actual_core,
        previous_goal_turn_classification='progress: actual train-fitted59/289majority and70/289copy-selection controls plus verified live-job wait',
        current_goal_turn_classification='progress:25actual CPU causal/framing invariants rule out the tested implementation failure modes without runtime changes',
        actual_parameterized_causal_cases_passed=24,actual_pinned_tokenizer_control_passed=True,
        causal_scope=dict(model_paths=['own dense backbone','own dense policy actor'],CPU_precisions=['FP32','BF16'],
            image_sides=[None,16,32],attention_paths=['all-valid fast causal','padded explicit causal'],training=True,gradient_checkpointing=True,
            future_token_changes_leave_past_logits_bit_exact=True,masked_past_keys_leave_real_logits_bit_exact=True,
            future_input_embedding_gradient_exactly_zero_and_past_gradient_nonzero=True,
            CPU_actor_frozen_cast_cache_actually_attached_for_BF16=True,all_test_model_parameter_values_and_grad_fields_unchanged=True),
        tokenizer_scope=dict(actual_files_byte_identical=True,postprocessor=None,automatic_truncation=None,automatic_padding=None,
            native_special_ids=[0,1,2,3],representative_coding_prompts_each_file=48,default_and_explicit_IDs_identical=True,complete_roundtrip_bytes_preserved=True),
        actual_packages=dict(torch=torch.__version__,tokenizers=tokenizers.__version__),
        small_explicit_own_neural_models_only=True,production_weights_not_loaded=True,CUDA_or_full_size_numerical_parity_not_tested=True,
        no_original_source_queue_or_training_setting_edited=True,new_production_optimizer_updates_by_check=0,
        no_fault_found_in_tested_CPU_mask_or_tokenizer_paths=True,cause_of_low_functional_correctness_not_established=True,
        actual_original_main=current,actual_live_processes=actual,
        actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        original_CPU_comparison=read(ROOT/'reports/full-function-repair-comparison-CPU-v1/progress.json'),
        original_interim_assessment=read(ROOT/'reports/public-pretraining-interim-CPU-v2/progress.json'),
        all_original17families_human_RL_and_nine_endpoint_scopes_unchanged=True,candidate_promoted=False,goal_achieved=False)
    atomic(OUT/'complete.json',proof)
    lines=['# Active runtime causality and tokenizer framing','',
        '**25actual controls pass** against source bytes that match the live main and queued actor runtimes. '
        'No original runtime, tokenizer, data, source mask, optimization setting, queue or neural worker is modified.', '',
        'The24causal cases cover both own dense and own policy-actor paths, CPU FP32/BF16, no image/base16px/doubled32px image inputs, '
        'and all-valid/padded attention masks with training mode and gradient checkpointing active. '
        'The actor BF16 cases actually attach the frozen own-matrix cast cache. '
        'Changing future input tokens leaves past logits bit exact. Changing a masked key in the past leaves every real-token logit bit exact. '
        'Gradients of an earlier prediction with respect to future input embeddings are exactly zero, while past gradients are nonzero. '
        'All parameters and parameter gradient fields remain unchanged.', '',
        'The two actual pinned training/evaluation tokenizer files are byte identical. Neither config enables a postprocessor, automatic truncation or automatic padding. '
        'The original pad/BOS/EOS/unknown IDs are0/1/2/3. All48representative task/language/prompt-view samples per tokenizer produce identical IDs '
        'with default and explicit special-token settings, and roundtrip to the whole original input text.', '',
        '**Scope:** these are structural CPU checks using small explicit own-neural configurations. Current production checkpoint weights are not loaded. '
        'The tests do not establish CUDA fused-kernel or full-size numerical parity, model quality, a throughput gain or the cause of low coding correctness. '
        'They find no causal-mask or hidden-tokenizer-framing fault in the tested paths. The evidence supports retaining the current implementation '
        'and waiting for actual matched learning/evaluation results before changing these mechanisms.', '',
        f'The original four-L40 main continues at **{len(rows):,}/4,447** updates and **{current["response_EOS_targets"]:,}** actual targets. '
        'The existing CPU comparison and interim native assessment remain live; the preference and reasoning warmups remain behind their exact predecessors. '
        'All17source families, human/DPO, execution RL, machine preferences and nine-endpoint full final assessments remain intact. '
        '**Latest complete local functional correctness remains0/48; no new accuracy improvement or frontier parity is established. All ten final requirements remain incomplete.**','']
    (OUT/'metrics.md').write_text('\n'.join(lines))
    review['requirements']['serious_full_model_training']['evidence']+=(
        ' Twenty-four actual own dense/actor CPU FP32/BF16 training/checkpointed causal controls pass with no/base/doubled images and all-valid/padded masks. '
        'Future token changes and masked past keys leave protected logits bit exact; future input-embedding gradients are0with nonzero past gradients. '
        'The two pinned tokenizers are byte identical with no postprocessor/truncation/padding, and48representative coding inputs per file have exact default/explicit IDs and roundtrips. '
        'Core tested files match every applicable live runtime binding. These small-graph CPU checks do not load production weights or prove CUDA/full-size parity or quality; no runtime change or new production update is made.')
    if len(review['requirements'])!=10 or any(value['status']!='incomplete_for_requested_final_model' for value in review['requirements'].values()):raise ValueError('Structural invariants complete none of the requested final-model capability requirements')
    review.update(at=at,previous_full_scope_review=str(previous),previous_goal_turn_classification=proof['previous_goal_turn_classification'],
        current_goal_turn_classification=proof['current_goal_turn_classification'],latest_actual_causal_and_framing_check=proof,
        source_files_sha256={**files,str(OUT/'complete.json'):sha(OUT/'complete.json')},goal_achieved=False)
    atomic(OUT/'review.json',review)
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Every tested and original bound source must remain exact before index update')
    audit.update(at=at,requirements=review['requirements'],latest_full_scope_review=str(OUT/'review.json'),latest_actual_main_runtime=current,
        latest_actual_causal_and_framing_check=str(OUT/'complete.json'),goal_achieved=False)
    atomic(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f'Causal/framing checks complete, **{at}**: **25actual controls pass** on source bytes matching the live main and actor runtimes. '
        'Own dense/actor CPU FP32/BF16 paths preserve past logits against future-token changes and masked past keys; future input gradients are exactly0. '
        'The pinned tokenizers are byte identical with no hidden framing/truncation/padding and exact representative input roundtrips. '
        '**These small-graph CPU checks find no tested masking/framing fault; they are not CUDA/full-size or model-quality evidence.** '
        f'The original main continues at **{len(rows)}/4,447**, with all source/human/RL/nine-endpoint scopes. '
        '**Latest complete functional correctness remains0/48; all ten final requirements remain incomplete.** '
        '[Exact checks and limits](reports/active-causal-and-framing-v1-review/metrics.md), '
        '[full review](reports/active-causal-and-framing-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_invariant_tests_passed=25,main_actual_updates=len(rows),new_production_optimizer_updates_by_check=0,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
