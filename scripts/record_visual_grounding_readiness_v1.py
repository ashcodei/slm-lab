"""Reconcile photo-prefix, own706M controls and paired-gradient implementation."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
import subprocess

import torch
from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.finalized_visual_pairs_v1 import FinalizedVisualPairs
from slm_lab.model import ModelConfig,TinyMultimodalLM
from slm_lab.vqa_pair_evaluation import text

ROOT=Path(__file__).resolve().parents[1]


def main():
    logs=[ROOT/'reports/paired-visual-contrast-v1-release-tests.log',ROOT/'reports/finalized-visual-pairs-v1-focused-tests.log']
    for log,count in zip(logs,[6,2]):
        if f'{count} passed' not in log.read_text() or 'FAILED' in log.read_text():raise ValueError('Current mathematical/source-interface checks required')
    source=Path('/tmp/pytest-of-jovyan/pytest-104/test_four_actual_workers_match0')
    retained=ROOT/'reports/paired-visual-contrast-v1-CPU'
    if retained.exists():raise ValueError('Inspect existing retained evidence rather than overwrite')
    ranks=[json.loads((source/f'rank-{i}.json').read_text()) for i in range(4)]
    if any(r['rank']!=i or not r['global_CE_and_eligible_contrast_gradients_match_independent_FP64_autograd'] for i,r in enumerate(ranks)):
        raise ValueError('Actual four-worker independent gradient evidence required')
    audit_path=ROOT/'reports/committed-paired-photo-audit-v1/results.json';audit=json.loads(audit_path.read_text())
    if not audit['passed'] or audit['training_admitted'] is not False:raise ValueError('Committed-prefix audit must not become partial admission')
    for row in audit['verified_original_files']:
        if file_hash(row['image'])!=row['image_sha256'] or file_hash(row['attribution'])!=row['attribution_sha256']:
            raise ValueError('Verified committed original bytes changed')
    diagnostic=ROOT/'reports/paired-image-dependence-CPU-v1';comparison=json.loads((diagnostic/'comparison.json').read_text())
    for path,digest in comparison['original_source_files_sha256'].items():
        if file_hash(path)!=digest:raise ValueError('Actual706M diagnostic source/model bytes differ')
    for name,summary in comparison['summaries'].items():
        actual=json.loads((diagnostic/(name+'.json')).read_text());rows=actual['results']
        if len(rows)!=4 or len({r['pair_id'] for r in rows})!=2:raise ValueError('Both full original image pairs required')
        correct=sum(text(r['proper']['generated'])==text(r['expected']) for r in rows)
        gap=sum(r['counterpart']['mean_answer_EOS_nll']-r['proper']['mean_answer_EOS_nll'] for r in rows)/4
        nll=sum(r['proper']['mean_answer_EOS_nll'] for r in rows)/4
        if (correct!=summary['proper_exact'] or not math.isclose(gap,summary['mean_image_conditioned_NLL_gap'],abs_tol=1e-12)
                or not math.isclose(nll,summary['mean_proper_answer_EOS_nll'],abs_tol=1e-12)):
            raise ValueError('Actual raw paired diagnostic aggregates differ')
    config=ModelConfig.load(ROOT/'configs/model.json');token=ROOT/'data/tokenizer-corpus/tokenizer.json'
    try:
        reader=FinalizedVisualPairs(ROOT,Tokenizer.from_file(str(token)),config,448)
        gate=dict(final_source_opened=True,original_pairs=len(reader),source_contract_sha256=reader.contract_sha256)
    except ValueError as error:
        if 'Finalized original real-photo pair manifest' not in str(error):raise
        gate=dict(final_source_opened=False,rejected_unfinalized_source=True,reason=str(error))
    with torch.device('meta'):counts=TinyMultimodalLM(config).parameter_counts()
    if counts['total']!=706012672 or counts['vision']!=51624448:raise ValueError('Exact own full-model parameter inventory required')
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True).stdout.splitlines();workers=[]
    for row in apps:
        pid=int(row.split(',')[0]);p=Path('/proc')/str(pid);fields=(p/'stat').read_text().split(') ',1)[1].split()
        argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();parent=int(fields[1])
        launcher_fields=(Path('/proc')/str(parent)/'stat').read_text().split(') ',1)[1].split()
        if (fields[0]=='Z' or 'slm_lab.training pretrain' not in argv or 'systems' not in argv
                or int(launcher_fields[1])!=2754743):
            raise ValueError('Exact original systems-code CUDA worker parentage required')
        workers.append(dict(pid=pid,parent_pid=parent,start_ticks=fields[19],argv=argv,GPU_record=row))
    if len(workers)!=4:raise ValueError('All four actual original GPU workers required')
    shutil.copytree(source,retained,ignore=shutil.ignore_patterns('rendezvous','*.lock'))
    metrics=[json.loads(x) for x in (ROOT/'reports/curriculum-disk-cleanup-resumption-v1/joint.log').read_text().splitlines() if x.startswith('{"step"')]
    code=[ROOT/'src/slm_lab/paired_visual_contrast_v1.py',ROOT/'src/slm_lab/finalized_visual_pairs_v1.py',
        ROOT/'tests/test_paired_visual_contrast_v1.py',ROOT/'tests/test_finalized_visual_pairs_v1.py',Path(__file__).resolve(),
        ROOT/'scripts/audit_committed_paired_photos_v1.py',ROOT/'scripts/diagnose_paired_image_dependence_v1.py']
    artifacts=logs+[audit_path,ROOT/'reports/committed-paired-photo-audit-v1/committed-prefix.jsonl']
    artifacts+=list(p for p in diagnostic.iterdir() if p.is_file())+list(p for p in retained.iterdir() if p.is_file())
    result=dict(format='own-paired-visual-grounding-readiness-v1',observed_at=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification=dict(classification='progress',reason='Repaired all86Python originals, preserved/admitted195-program374-pair Java/Python union and wired v6;15checks include real374-reference/twoCPU-update exact recovery.'),
        current_turn_classification=dict(classification='progress',reason='Verified1640committed real-photo credits and224/448pxpixels; executed actual two-checkpoint706M CPU image controls; implemented full proper CE plus eligible bidirectional image contrast and finalized-only source factory;8focused checks pass.'),
        committed_photo_prefix=dict(photos=1640,full_source_admitted=False,all_attribution_and224_448pixels_verified=True,
            candidate_pair_availability_before_final_screens=audit['candidate_pairs_with_both_committed_photos_before_final_screens'],
            high_consensus_binary_before_final_screens=audit['high_consensus_binary_candidate_pairs_before_final_screens']),
        actual_own706M_CPU_diagnostic=dict(checkpoints=comparison['summaries'],original_train_candidate_pairs=2,questions=4,
            precision='fp32',image_size=448,max_new_tokens=8,source_unfinalized=True,model_updates=0,
            independent_final_benchmark=False,capability_gain_claim=False),
        objective=dict(name='whole-answer CE plus bidirectional proper-versus-counterpart image NLL ranking',
            all_original_proper_CE_pairs_preserved=True,extra_contrast_only_opposite_yes_no_with_atleast9of10votes=True,
            same_question_and_answer_tokens_and_image_resolution_across_contrast=True,human_preference_RLHF=False,
            independent_FP64_global_gradient_oracle=True,vision_only_toy_update_text_only_logits_and_all_language_tensors_identical=True,
            zero_eligible_pairs_keep_all_proper_CE=True,full_vision_only_training_loop_and_replay_not_yet_integrated=True,
            existing_model_counts=counts,added_model_parameters=0,actual_full706M_grounding_updates=0),
        checks=dict(total=8,mathematics_and_four_workers=6,source_interface224_448fixtures=2,
            actual_four_CPU_workers=ranks,real_final_source_factory_gate=gate,
            source_interface_positive_cases_are_explicit_admission_dependency_fixtures=True),
        research_sources=['https://proceedings.neurips.cc/paper/2019/hash/51d92be1c60d1db1d2e5e7a07da55b26-Abstract.html',
            'https://openaccess.thecvf.com/content/CVPR2021/html/Niu_Counterfactual_VQA_A_Cause-Effect_Look_at_Language_Bias_CVPR_2021_paper.html',
            'https://aclanthology.org/2024.findings-emnlp.775/','https://arxiv.org/abs/2608.12158'],
        research_scope='Motivation only; local source-derived CE/ranking differs from published DPO/causal methods. No external weights, teacher generations, replication or SOTA claim.',
        actual_original_training=dict(status=json.loads((ROOT/'reports/dialogue-retention-v1/status.json').read_text()),
            workers=workers,latest_metric=metrics[-1],new_visual_contrast_used=False),
        remaining_requirements=['Finalize original paired photos and all30-source/full-base coverage; preserve every original source and genuine-feedback continuation.',
            'Integrate vision-only candidate loop, original visual replay, full checkpoints and paired/caption/OCR/object/domain/coding retention evaluations before any promotion.',
            'Complete actual full-model capacity/training and unchanged broad capability assessments; existing native correctness and visual grounding remain weak.'],
        candidate_promoted=False,new_model_capability_gain_measured=False,high_end_capability_established=False,goal_achieved=False,
        files_sha256={str(p):file_hash(p) for p in code+artifacts})
    path=ROOT/'reports/paired-visual-grounding-v1-readiness.json';atomic_json(path,result)
    print(json.dumps(dict(report=str(path),committed_photos=1640,actual_model_diagnostics=comparison['summaries'],
        focused_checks=8,actual_original_step=metrics[-1]['step'],real_final_source_factory_gate=gate,goal_achieved=False),indent=2))


if __name__=='__main__':main()
