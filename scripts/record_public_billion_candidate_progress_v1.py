"""Record source-scale progress and actual partial coding results without promotion."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/public-billion-candidate-progress-v1-review'
STAGES = ['code-first-continuation-v3', 'phase-human-preference-v7', 'broad-anchored-code-v6',
    'code-first-final-standalone-CPU-v3', 'full-held-QA-pipeline-v4', 'full-AI2D-pipeline-v4',
    'full-original-MBPP-pipeline-v4', 'AI2D-fixed-exposure-strata-v4', 'full-machine-preference-v1',
    'machine-preference-final-standalone-CPU-v1', 'machine-standalone-full-assessment-v1']


def read(path):return json.loads(Path(path).read_text())


def live(identity):
    if terminal(identity):raise ValueError('Every protected exact original process birth must remain live')
    fields = (Path('/proc')/str(identity['pid'])/'stat').read_text().split(') ', 1)[1].split()
    return dict(identity, state=fields[0])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('Fresh GPU-hidden actual full-scope progress review required')
    master_path = ROOT/'reports/completion-audit.json'
    master = read(master_path)
    previous_path = ROOT/master['latest_full_scope_review']
    old_review = read(previous_path)
    protected = set()
    controllers = []
    for name in STAGES+['public-billion-causal-candidate-CPU-v2', 'public-billion-causal-candidate-v2-full-readback-v1']:
        path = ROOT/'reports'/name/'plan.json'
        if not path.exists():path = ROOT/'reports'/(name+'-queue.json')
        plan = read(path)
        protected.update(plan['source_files_sha256'])
        controllers.append(dict(stage=name, owner=live(plan['owner'])))
    neural = ROOT/'reports/code-first-step2048-native-code-CPU-v1'
    neural_plan = read(neural/'plan.json');protected.update(neural_plan['source_files_sha256'])
    coding_owner = live(neural_plan['owner'])
    native_reader = read(ROOT/'reports/code-first-step2048-native-code-CPU-v1-readback-queue.json')
    coding_reader_owner = live(native_reader['owner']);protected.update(native_reader['source_files_sha256'])
    source = ROOT/'reports/code-first-continuation-v3'
    progress = read(source/'progress.json')
    children = [p for p in read(source/'children.json') if p['subject'].startswith('train-rank-')]
    if len(children) != 4 or (source/'failure.json').exists():raise ValueError('The original four-L40 training must remain healthy')
    workers = [live(p) for p in children]
    actual_pids = {int(p.strip()) for p in subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True).splitlines() if p.strip()}
    if actual_pids != {p['pid'] for p in workers}:raise ValueError('Only the four exact original workers may own the GPUs')
    source_folder = ROOT/'reports/public-billion-causal-candidate-CPU-v2'
    source_progress = read(source_folder/'progress.json')
    if source_progress['candidate']['train_target_tokens_by_quota']['code:Go'] < 1 or (source_folder/'failure.json').exists():
        raise ValueError('Real original GO rows must actually reach the corrected quota without a new source failure')
    prefix_path = ROOT/'reports/public-billion-native-prefix-CPU-v1/complete.json'
    prefix = read(prefix_path)
    if (not prefix['complete'] or prefix['actual_original_documents'] != 1024 or len(prefix['actual_code_languages']) != 8
            or prefix['optimization_admitted'] is not False or prefix['actual_independently_rebuilt_prefix']['actual_train_targets'] != 4047848):
        raise ValueError('The actual1024-document original/native/causal-window prefix must finish without false admission')
    handoff_path = ROOT/'reports/public-billion-source-language-handoff-v2/complete.json'
    handoff = read(handoff_path)
    if not handoff['complete'] or not terminal(handoff['original_owner']):raise ValueError('The verified-bug CPU collector handoff must finish')
    scale_path = ROOT/'reports/recent-pretraining-scale-CPU-v1/complete.json'
    scale = read(scale_path)
    if (not scale['complete'] or not scale['all_model_ancestry_token_exposure_not_reconstructed']
            or scale['completed_main_actual_exposure']['total']['response_EOS_targets'] != 467581587
            or scale['full_current_declared_exposure']['response_EOS_targets'] != 105260572):
        raise ValueError('Actual recent source-target accounting must retain its limited historical scope')
    loss_paths = [source/('held-packed-'+n+'.json') for n in ['initial', '2048', '4096']]
    losses = [read(path) for path in loss_paths]
    controls = [{k:v for k,v in row.items() if k != 'token_mean_NLL'} for row in losses]
    if controls[0] != controls[1] or controls[0] != controls[2] or losses[0]['complete_causal_targets'] != 524160:
        raise ValueError('Every reported token-loss comparison must use identical original inputs and denominators')
    loss_reduction = 100*(losses[0]['token_mean_NLL']-losses[2]['token_mean_NLL'])/losses[0]['token_mean_NLL']
    initial = read(neural/'initial/completion-complete.json')['actual_results']
    trained = read(neural/'step2048/completion-complete.json')['actual_results']
    if initial['original_native_tasks'] != 24 or trained['original_native_tasks'] != 24:
        raise ValueError('Only a complete matched24-task frame may be reported as such')
    tests = {'public-pretraining-intake-v2-tests':3, 'public-source-sigint-v2-tests':1,
        'public-causal-windows-and-billion-readback-v1-final-tests':7, 'public-billion-full-reader-driver-v1-final-tests':2}
    for name, count in tests.items():
        if str(count)+' passed' not in (ROOT/'reports'/(name+'.log')).read_text():raise ValueError('Every targeted control must pass')
    evidence_text = {
        'own_architecture':'Own scratch-origin dense text backbone, original BPE tokenizer, image encoder and byte/native word heads remain. The new source/causal-window pipeline adds no neural parameters or external weights. Full-code, original execution-output/state supervision and retention replay are active; chosen-code, human DPO and execution-RL comparisons remain scheduled. No globally novel architecture or useful model gain is established by this data work.',
        'public_data':f'The completed main uses28declared source families. A separate corrected collector is live with{source_progress["candidate"]["actual_train_targets"]}retained train targets toward1B:500Mpublic per-file-license-filtered GitHub code across8languages,250Mpublic web math and250Meducational web. Original pinned public Parquet bytes and source fields are verified; web-page rights are not relicensed by catalog licenses. The whole1024-document code prefix reconstructs4,047,848train targets and15,018held targets across2,290original observed rows, including39original GO rows. The complete billion-target corpus is not yet collected, independently admitted or trained. Broad domain balance and semantic/all-history independence are not certified.',
        'under_1b_parameters':'The active complete dense student has984,399,690unique values including all text/image/word and auxiliary heads. The separate actor branches remain below1B; reference policies and optimizer states are training resources, not exported student parameters. The new raw-causal data path changes no model values or final parameter count.',
        'text_and_images':'Generic photos, visual QA, caption matching, OCR and real-word pathways remain in the original source/replay/assessment scope. Latest completed main development checks remain weak: OCR exact0/20 and object classification3/15. Full3,088-question AI2D under proper/blank/wrong images and3,429held word crops under3conditions remain scheduled; word crops are not full-photo text detection. The new1Bcollection is text/code/math pretraining data and establishes no new image gain.',
        'human_feedback':'The original complete85,786-public-human-pair chosen-CE/DPO comparison and4,642held-pair assessment remain queued. Execution-reward RL on all3,721original sources and the later1,620-pair machine-preference comparison remain separate from human feedback. Public human rankings of other-model answers are not new human judgments of student outputs. No new production human or machine preference update is claimed here.',
        'serious_full_model_training':f'The complete35,284-update main is finished. The original four-L40 code/state/replay continuation is live at{progress["actual_updates"]}/7,558updates, updating all selected full model groups with original recovery/source/cold-load controls. The recent source audit reconstructs467,581,587main supervised targets and105,260,572planned continuation targets; these are repeated target presentations from two recent runs, not all ancestral training or unique-document exposure. Undertraining is a supported hypothesis, not a completed causal ablation. The new1Bsource candidate has0neural updates and needs complete source readback, a concrete fresh GPU training driver and capacity/recovery checks.',
        'high_end_llm_capability':f'NOT ACHIEVED. Identical128held code windows/524,160targets have producer NLL{losses[0]["token_mean_NLL"]:.9f}initial,{losses[1]["token_mean_NLL"]:.9f}at2,048and{losses[2]["token_mean_NLL"]:.9f}at4,096: {loss_reduction:.4f}%lower than the start. The step2,048state/input reader is complete; step4,096numbers are original producer observations with statically identical input metadata, not a new independent neural run. The completed matched24-task completion frame remains0functional/empirical-efficiency passes at both endpoints, syntax12to10. No frontier, all-round or arbitrary-code capability gain is established.',
        'broad_independent_benchmarks':'All original8fixed lineage endpoints retain16,961QA questions,3,088visual questions under3conditions and600original MBPPproblems/1,816original cases. Later machine-arm and standalone comparisons retain these populations plus4,642held human pairs,3,429held word crops under3conditions, and unchanged native/broad checks. These controllers are still waiting. Current24-task completion results are local development observations; the step2,048instruction frame and96-delivery independent source/outcome readback remain live. Declared held stems are screened lexically in new actual native data, not certified independent of every ancestral source or semantically uncontaminated.',
        'gpu_runtime':f'The same four original NVIDIA L40workers are live; all11original pipeline controllers are preserved. Only the verified GO-spelling-bug CPUcollector was retired by exact pidfdSIGINT, preserving196,481,825partial unadmitted train targets and every original artifact. A tested compatibility path handled this Python build having no os.pidfd_open wrapper; the failed first handoff log/receipt are preserved. Corrected collection and the full independent reader are CPU-hidden, not GPUowners. Thirteen new targeted controls and the actual1024-document independent source/native/window proof pass. No GPU training/evaluation process was signalled or restarted.',
        'any_code_correctness_and_efficiency':'NOT ACHIEVED. The actual step2,048completion frame scores0/24functional and0/24empirical-efficiency passes, versus0/24at its own initial endpoint; syntax12/24to10/24and ordinaryPython0/12at both. Original raw generated actions and actual fresh native/ordinary interpreter outcomes are retained; programs are not repaired. The instruction frame is still being generated. Existing code/state supervision, public case reward and DPO remain bounded experiments; finite cases do not prove arbitrary-code correctness or asymptotic efficiency. The source fix and token-loss reduction are not coding-accuracy improvements.'}
    if set(evidence_text) != set(old_review['requirements']):raise ValueError('Every original final requirement must remain in full scope')
    requirements = {k:dict(status='incomplete_for_requested_final_model', evidence=v) for k,v in evidence_text.items()}
    at = datetime.now(timezone.utc).isoformat()
    runtime = dict(at=at, actual_code_first_updates=progress['actual_updates'], planned_code_first_updates=7558,
        actual_original_GPU_workers=workers, all_thirteen_live_pipeline_and_source_controllers=controllers,
        actual_CPU_coding_owner=coding_owner, actual_CPU_coding_reader_owner=coding_reader_owner,
        actual_CPU_coding_progress=read(neural/'progress.json'), actual_free_persistent_bytes=shutil.disk_usage(ROOT).free,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used',
            '--format=csv,noheader'], text=True), original_GPU_training_and_all_eleven_previous_controllers_preserved=True)
    docs = [ROOT/'README.md', ROOT/'RESEARCH.md', master_path]
    if any(str(p) in protected for p in docs):raise ValueError('No live scientific source-bound document may be rewritten')
    OUT.mkdir()
    atomic_json(OUT/'runtime.json', runtime)
    atomic_json(OUT/'captured-source-progress.json', source_progress)
    atomic_json(OUT/'captured-training-progress.json', progress)
    evidence = [Path(__file__), previous_path, scale_path, prefix_path, handoff_path, *loss_paths,
        neural/'initial/completion-complete.json', neural/'step2048/completion-complete.json',
        ROOT/'reports/public-pretraining-source-preflight-CPU-v1-readback/complete.json',
        ROOT/'reports/public-billion-causal-candidate-v2-full-readback-v1/plan.json',
        source_folder/'plan.json', OUT/'runtime.json', OUT/'captured-source-progress.json', OUT/'captured-training-progress.json',
        *[ROOT/'reports'/(name+'.log') for name in tests]]
    atomic_json(OUT/'review.json', dict(at=at, requirements=requirements, actual_runtime=runtime,
        previous_full_scope_review=str(previous_path.relative_to(ROOT)),
        source_files_sha256={str(p):file_hash(p) for p in evidence},
        previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True, original_scope_and_historical_evidence_preserved=True,
        new_source_training_updates=0, candidate_promoted=False, goal_achieved=False))
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at, requirements={k:v['evidence'] for k,v in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)), latest_actual_main_runtime=runtime,
        current_billion_public_source_candidate=str((source_folder/'plan.json').relative_to(ROOT)),
        current_billion_public_independent_reader='reports/public-billion-causal-candidate-v2-full-readback-v1/plan.json',
        latest_actual_public_native_prefix='reports/public-billion-native-prefix-CPU-v1/complete.json',
        next_required_work=[
            'Finish the same7558-update code/state/replay run and96-delivery CPU coding comparison/readback; assess actual correctness before promotion.',
            'Complete every declared1Bsource quota, then the queued independent original-source/native/held/rights/disposition readback and4096window data admission. Do not train a partial candidate.',
            'Implement and test a concrete own-model raw-pretraining branch using the admitted full corpus, balanced existing image/word/broad retention, fresh four-L40capacity/private-RNG/recovery/cold-load controls and explicit GPU ownership. Existing scientific jobs and lineage stay intact.',
            'Finish full original human preference and execution-RL arms, machine comparison, standalone exports and full original QA/visual/code/human/word assessments. Independently verify actual final-model gains across all ten original requirements.'],
        current_goal_turn_classification='progress', all_ten_final_requirements_incomplete=True,
        candidate_promoted=False, goal_achieved=False)
    atomic_json(master_path, master)
    paragraph = (f'Public pretraining scale and real coding result, **{at}**: original four-L40 training continues at '
        f'**{progress["actual_updates"]}/7,558 updates**. Matched128-window code NLL is **0.868549 → 0.825648 '
        f'({loss_reduction:.2f}% lower)** at step4,096, while the completed step2,048completion frame remains '
        '**0/24 functional passes** (syntax12→10); its instruction frame is still running. Lower token loss is not a correctness gain.\n\n'
        f'A separate **1B-target public code/math/web candidate** is collecting; the captured prefix contains '
        f'**{source_progress["candidate"]["actual_train_targets"]:,} train targets** and is **not admitted or trained**. '
        'The publisher GO spelling fix is tested and live. The partial old collector is preserved; all original GPU jobs remain unchanged. '
        '**1,024 actual original files / 4,047,848 train targets** across8coding languages independently rebuild '
        'through source bytes, native tokens, roles, rights, all observed dispositions and bounded4096token windows. '
        'A full independent source reader is queued after collection; **13 new controls pass**. '
        '**All ten final requirements remain incomplete.** '
        '[Recent exposure audit](reports/recent-pretraining-scale-CPU-v1/complete.json), '
        '[actual native prefix](reports/public-billion-native-prefix-CPU-v1/complete.json), '
        '[complete review](reports/public-billion-candidate-progress-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for path in docs[:2]:
        first, rest = path.read_text().split('\n', 1)
        path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'), actual_code_first_updates=progress['actual_updates'],
        actual_retained_unadmitted_source_targets=source_progress['candidate']['actual_train_targets'],
        latest_completed_coding_frame_functional=trained['native_functionally_correct'],
        new_source_training_updates=0, all_ten_final_requirements_incomplete=True, goal_achieved=False)), flush=True)


if __name__ == '__main__':main()
