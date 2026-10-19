"""Record complete source reconstruction, live full controls and unfinished goals."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import read, sha, atomic
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/full-human-token-suffix-curriculum-v1-review'
PREVIOUS = ROOT / 'reports/first-repair-quality-and-cached-visual-runtime-v1-review/review.json'


def main():
    if OUT.exists():
        raise ValueError('Preserve every historical completed and failed result')
    checks = [('public-token-suffix-curriculum-v1-tests.log', 10), ('independent-token-suffix-readback-v2-tests.log', 8)]
    for name, count in checks:
        if str(count) + ' passed' not in (ROOT / 'reports' / name).read_text():
            raise ValueError('All original target/role/Unicode/context and independent source controls must pass')
    metadata_path = ROOT / 'reports/full-public-token-suffix-metadata-readback-CPU-v1/complete.json'
    metadata = read(metadata_path)
    if (not metadata['passed'] or metadata['actual_original_human_sources'] != 3721
            or metadata['actual_all_four_native_views'] != 14884
            or not metadata['all3721original_source_roles_inputs_masks_reference_bytes_unicode_prefixes_suffixes_and_EOS_budgets_rebuilt']
            or metadata['source_training_admitted'] or metadata['model_updates'] != 0):
        raise ValueError('All actual source/target/byte reconstruction must complete without claiming RL admission')
    expected_missing = {'4': 18606, '16': 62985, '64': 208629, '0': 584218}
    expected_hints = {'4': 3719, '16': 3660, '64': 2344, '0': 0}
    for level, value in metadata['actual_per_level_totals'].items():
        if (value['views'] != 3721 or value['original_context_tokens'] != 2287522
                or value['original_response_EOS_targets'] != 584218
                or value['actual_missing_response_EOS_targets'] != expected_missing[level]
                or value['actual_hint_views'] != expected_hints[level]):
            raise ValueError('No original source or unconditional CE/EOS target can be lost to easier prefix levels')
    producer_plan = ROOT / 'reports/full-public-token-suffix-qualification-CPU-v1/plan.json'
    reader_plan = ROOT / 'reports/full-public-token-suffix-readback-CPU-v1/plan.json'
    remaining_plan = ROOT / 'reports/remaining-GPU-repair-native-program-readback-CPU-v1-wait/plan.json'
    producer, reader, remaining = map(read, [producer_plan, reader_plan, remaining_plan])
    for plan in [producer, reader, remaining]:
        if terminal(plan['owner']):
            raise ValueError('Reassess the actual terminal/completion state before using this live-work review')
    if producer['owner'] != dict(pid=146084, ticks='55111340') or reader['exact_original_source_producer'] != producer['owner']:
        raise ValueError('The independent reader must wait on the exact original source-only producer')
    if any(sha(p) != h for p, h in producer['source_files_sha256'].items()):
        raise ValueError('All frozen full-source producer inputs and code must remain unchanged')
    study = ROOT / 'reports/full-function-repair-comparison-GPU-v2'
    semantic = read(study / 'semantic_repair/complete.json')
    if not semantic['complete']:
        raise ValueError('This review expects the original complete second endpoint')
    summaries = {}
    for profile in ['baseline', 'plain_repeat', 'semantic_repair']:
        reports = [read(study / profile / ('code-' + frame + '.json')) for frame in ['completion', 'instruction']]
        summaries[profile] = dict(original_code_programs=sum(r['task_count'] for r in reports),
            syntax=sum(r['by_language'][lang]['syntax_passed'] for r in reports for lang in ['Python', 'JavaScript']),
            functional=sum(r['by_language'][lang]['functional_passed'] for r in reports for lang in ['Python', 'JavaScript']),
            empirical_efficiency=sum(r['by_language'][lang]['efficiency_passed'] for r in reports for lang in ['Python', 'JavaScript']),
            independently_fresh_program_checked=profile in ['baseline', 'plain_repeat'])
    at = datetime.now(timezone.utc).isoformat()
    def process(plan):
        owner = plan['owner']
        maps = Path('/proc', str(owner['pid']), 'maps').read_text()
        fds = []
        for fd in Path('/proc', str(owner['pid']), 'fd').iterdir():
            try:
                fds.append(str(fd.resolve()))
            except FileNotFoundError:
                pass
        return dict(owner=owner, actual_birth_live=not terminal(owner), actual_scheduler=os.sched_getscheduler(owner['pid']),
            actual_libtorch_mapped='libtorch' in maps, actual_libcuda_mapped='libcuda.so' in maps,
            actual_shared_GPUlease_FD_present=any('code-residual-production-v4.gpu.lock' in p for p in fds))
    snapshot = dict(at=at, actual_source_producer=process(producer), actual_full_independent_source_reader=process(reader),
        actual_remaining_native_code_reader=process(remaining),
        original_source_producer_progress=read(producer_plan.parent / 'progress.json'),
        original_remaining_GPUchildren=[p for p in read(study / 'children.json') if not terminal(p)],
        original_complete_code_endpoint_summaries=summaries,
        original_complete_endpoints=[p.parent.name for p in study.glob('*/complete.json')],
        actual_original_GPUparent_live=not terminal(dict(pid=101066, ticks='54137054')),
        actual_visual_preparation_live=not terminal(dict(pid=136092, ticks='54738417')),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True), actual_free_bytes=os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize,
        original_visual_pixel_reader_progress=read(ROOT / 'reports/full-multimodal-pixel-readback-CPU-v1/progress.json'),
        no_existing_neural_worker_signalled_reprioritized_restarted_or_patched=True,
        no_new_GPU_context_lease_or_job=True, no_mutable_progress_journal_registry_or_human_queue_bound=True)
    requirements = deepcopy(read(PREVIOUS)['requirements'])
    requirements['public_data']['evidence'] = (
        'All3721existing original public human training programs(173documented/3548APPS) now have4native token-suffix contexts: '
        'at least4/16/64missing non-EOS tokens and the original fully unhinted prompt. An independent complete source reader '
        'rebuilds all14884views from original roles, whole input/label bytes, Unicode-exact prefixes, suffixes, trueEOS and '
        'uncropped native budgets. Every level retains all2287522original context tokens and584218unconditional response/EOS '
        'targets. Actual remaining suffix targets total18606/62985/208629/584218acrosslevels. No outside pretrained weights, '
        'teacher API or model-generated code is introduced by this new curriculum. Every source and failed/trivial/abstaining '
        'disposition remains; all17original public families and the separate unadmitted public synthetic candidate remain unchanged.')
    requirements['human_feedback']['evidence'] = (
        'The new curriculum is an outcome-verified code-learning proposal using original human public program demonstrations; '
        'it is not human preference feedback or a completed RLHF milestone. The exact live source-only producer attempts all3721 '
        'full references and14884prefix-then-EOS controls; an exact independent queued reader will reconstruct every view and '
        'freshly execute all18605controls again. Views passing with no completion, short sources without a real hint, unsupported '
        'reward sources, embedded boundary tokens, abstentions and fresh disagreements cannot silently gain conditional RL '
        'eligibility. Alloriginal unconditional CE targets remain. Source/productionRLadmission, own fresh rollouts, gradient/'
        'recovery/full-size GPUcapacity/training and unhinted quality are pending. All85786human preference pairs and canonical '
        'human/RLscopes remain intact.')
    requirements['serious_full_model_training']['evidence'] = (
        'All3actual four-L40coding arms remain trained at306updates/4851presentations/2287110response-EOS targets each, with '
        'independent source/Adam/readback. Baseline/first/second complete local endpoints exist; the third endpoint is still '
        'evaluating. New source controls and view reconstruction create0neural forwards/production updates and do not count '
        'as model improvement. Future suffix learning must preserve all3721sources and original CE/EOS targets alongside its '
        'conditional objective. The complete visual4733step/302627view epoch, full human/RLstages and9final endpoints remain.')
    requirements['any_code_correctness_and_efficiency']['evidence'] = (
        'Last independent baseline/first-arm results remain syntax14/48to21/48,functional1/48to1/48,empirical efficiency0/48to1/48 '
        'andordinaryPython0/24to1/24, with the Python sum_even gain and JavaScript lower_bound loss both explicit. The complete '
        'second producer endpoint reports30/48syntactic,1/48functional,1/48empirically efficient; fresh program confirmation for '
        'the second/third arms is pending. A source-only reader is verified live waiting for the exact original study birth '
        'to finish and restore, then freshly executes both remaining complete native/ordinary/source program populations and '
        'combines them with the immutable first readback. Every original192native/96ordinaryPython/60source program is required. '
        'No winner, arbitrary-code competence, broad-quality gain or new curriculum learning gain is claimed.')
    requirements['gpu_runtime']['evidence'] = (
        'The exact original GPUstudy remains live evaluating its final matched arm. The new full3721source qualification is '
        'CUDA-hidden,nice19,one-thread,SCHED_IDLE, with bounded output and40GiBother-writer reserve. Its fresh reader and remaining '
        'code reader wait source-only without model/GPUlease. No existing neural process is signalled, reprioritized, restarted '
        'or patched. The prepared full cached visual GPUv2runner remains unlaunched and still requires4actual full-size '
        'gradient/Adam/RNG-equivalence prefixes, every original recovery/cold gate, exact original aligned/source predecessors, '
        'fresh normalnice0launch and original exclusive lease. No new CUDAcapacity, speed or quality result is inferred.')
    for value in requirements.values():
        value['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10 or any(v['status'] != 'incomplete_for_requested_final_model' for v in requirements.values()):
        raise ValueError('All ten requested final-model requirements remain incomplete')
    names = ['src/slm_lab/public_token_suffix_curriculum_v1.py', 'src/slm_lab/independent_token_suffix_readback_v1.py',
        'tests/test_public_token_suffix_curriculum_v1.py', 'tests/test_independent_token_suffix_readback_v1.py',
        'tests/test_independent_token_suffix_readback_v2.py', 'scripts/run_public_token_suffix_curriculum_tests_v1.py',
        'scripts/run_independent_token_suffix_readback_tests_v1.py', 'scripts/run_independent_token_suffix_readback_tests_v2.py',
        'scripts/qualify_full_public_token_suffixes_CPU_v1.py', 'scripts/read_full_public_token_suffixes_CPU_v1.py',
        'scripts/read_all_token_suffix_metadata_CPU_v1.py', 'scripts/read_remaining_GPU_repair_endpoint_programs_CPU_v1.py',
        'reports/independent-token-suffix-readback-v1-tests.log',
        'reports/first-GPU-repair-native-program-readback-CPU-v1/complete.json',
        'reports/function-termination-full-CPU-v1-readback/complete.json',
        'reports/conditional-prefix-full-CPU-v1-endpoint-readback/complete.json']
    files = {str(p): sha(p) for p in [Path(__file__), PREVIOUS, metadata_path, producer_plan, reader_plan, remaining_plan,
        *[ROOT / 'reports' / n for n, _ in checks], *[ROOT / n for n in names], study / 'semantic_repair/complete.json']}
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    references = dict(checked_at=at,
        primary_research=[dict(title='Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning',
            authors='Xi et al.', year=2024, url='https://proceedings.mlr.press/v235/xi24a.html',
            relevance='Published demonstration-prefix curriculum motivates starting near a solution end and moving toward the original prompt.'),
            dict(title='RL for Reasoning by Adaptively Revealing Rationales', authors='Amani et al.', year=2026,
                url='https://proceedings.iclr.cc/paper_files/paper/2026/hash/c073a9d002f3dab99a7af18642d74bdb-Abstract-Conference.html',
                relevance='Published AdaBack varies the revealed prefix per sample from past reward; this implementation does not yet adapt prefixes from student rewards.'),
            dict(title='Backplay: Man muss immer umkehren', authors='Resnick et al.', year=2018,
                url='https://arxiv.org/abs/1807.06919', relevance='Earlier reverse demonstration curricula for sparse-reward reinforcement learning.')],
        this_is_an_implementation_adaptation_not_a_claim_to_invent_reverse_curriculum=True,
        local_hypothesis_not_an_observed_student_result='Shorter missing human-code suffixes may improve reward density; exact per-view execution rejects zero-work full-credit prefixes before a later sampled RL test.',
        local_changes=['4/16/64missing-token and fully unhinted views with exact original native token bytes',
            'all3721human source CE anchors retained at all4levels', 'full-reference and prefix-only controls for every source/view',
            'separate independent reconstruction plus every control freshly reexecuted', 'no model/API teacher or supplied training prefix in final unhinted benchmarks'],
        no_published_paper_result_is_a_score_for_this_student=True)
    atomic(OUT / 'research-context.json', references)
    classification = ('progress:all3721human sources/all14884token-suffix views independently reconstruct every original CE/EOS '
        'target; live full18605control producer and separate full fresh reader launched;18source controls pass; final original '
        'GPUarm and remaining192program comparison readback continue; no new learning or quality gain')
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        actual_original_human_sources=3721, actual_native_views=14884, actual_source_controls=18,
        actual_all14884view_source_and_target_reconstruction_passed=True,
        unconditional_context_tokens_retained_per_level=2287522, unconditional_response_EOS_targets_retained_per_level=584218,
        actual_remaining_suffix_EOS_targets_per_level=expected_missing, actual_hint_views_per_level=expected_hints,
        full_reference3721_and_prefix14884_execution_producer_verified_live=True,
        full_independent18605fresh_execution_reader_verified_live=True,
        final_original_study_program_readback_waiter_verified_live=True,
        all_failed_trivial_abstaining_and_disagreeing_cases_preserved=True,
        current_source_execution_progress_is_not_a_student_quality_score=True,
        second_coding_endpoint_producer_summary=summaries['semantic_repair'], second_endpoint_independent_program_readback_pending=True,
        no_new_neural_generation_training_update_GPUcontext_or_GPUlease=True,
        new_source_training_admitted=False, new_production_RL_admitted=False,
        student_reward_density_and_unhinted_quality_not_measured=True,
        original_code_visual_human_and_nine_endpoint_scopes_preserved=True,
        candidate_promoted=False, all_ten_final_requirements_incomplete=True, goal_achieved=False))
    for p in [OUT / 'complete.json', OUT / 'runtime-snapshot.json', OUT / 'research-context.json']:
        files[str(p)] = sha(p)
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, previous_full_scope_review=str(PREVIOUS),
        previous_goal_turn_classification='progress:first coding arm fresh quality plus38cached-runtime controls/full GPUv2implementation',
        current_goal_turn_classification=classification, source_files_sha256=files, goal_achieved=False))
    metrics = ('The next code-learning approach starts with short missing suffixes of **original public human code** and '
        'progresses toward the original unhinted problem. It adapts published reverse-curriculum ideas, rather than claiming '
        'the underlying idea is new: [R³,ICML2024](https://proceedings.mlr.press/v235/xi24a.html) and '
        '[AdaBack,ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c073a9d002f3dab99a7af18642d74bdb-Abstract-Conference.html). '
        'The current4/16/64/unhinted levels are fixed; no adaptive student-reward schedule is implemented or claimed.\n\n'
        '**Complete independent source reconstruction passes:**3721original human programs(173documented/3548APPS), '
        '14884views, exact original roles/native input IDs/response bytes/masks/Unicode prefixes/complete suffixes/trueEOS. '
        'Every level retains all**2287522original context tokens and584218unconditional CE/EOS targets**.\n\n'
        '| Nominal missing non-EOS tokens | Whole views | Actual hinted views | Remaining suffix/EOS targets |\n'
        '|---|---:|---:|---:|\n|4|3721|3719|18606|\n|16|3721|3660|62985|\n|64|3721|2344|208629|\n'
        '|Original unhinted problem|3721|0|584218|\n\n'
        'Shorter references without a real hint retain allCEtargets and cannot silently qualify as conditional hinted RL. '
        'The original failed independent test expected every nominal level to have a hint; the corrected test checks the '
        'actual-hint requirement. The failure log/source remain preserved, with no producer/reconstruction rule relaxed. '
        '**18source checks pass.**\n\n'
        'The exact live source-only producer now attempts **3721full original references and14884prefix-then-EOS controls**. '
        'The separate live independent reader waits for that exact birth, then rederives allviews and freshly attempts '
        '**all18605controls again**. Prefixes earning full credit without a completion are rejected; every failed, unsupported, '
        'abstaining or disagreeing case is preserved. One early non-credit reference is explicitly an uninformative reward '
        'source, not a discovered incorrect human solution. **These are counterfactual source controls, not student rollouts '
        'or a reward-density/quality score. Source/productionRLadmission, actual sampling/training/full-size GPUcapacity '
        'and unhinted improvement remain pending.**\n\n'
        'The original second coding endpoint is complete and reports **30/48syntax,1/48functional,1/48empirically efficient**; '
        'its fresh program readback is pending. The final arm remains under evaluation. A new source-only waiter will check '
        'the two remaining complete arms after the exact original GPUstudy finishes/restores, then combine the immutable '
        'first readback so all192native/96ordinaryPython/60source programs are covered. The last independently verified '
        'overall functional result remains **1/48→1/48**, with Python0/24→1/24 and one JavaScript pass lost. No winner or broad '
        'gain is claimed.\n\n'
        'No existing neural worker, queue, initializer, dataset, held evaluation or GPUlease was changed. The full cached '
        'visual GPUv2runner remains unlaunched. All17public families,85786human preference pairs, full visual/human/RLstages '
        'and9final endpoints remain. **All ten requested final requirements remain incomplete.**\n')
    (OUT / 'metrics.md').write_text(metrics)
    audit_path = ROOT / 'reports/completion-audit.json'
    audit = read(audit_path)
    audit.update(at=at, updated_at=at, requirements=requirements, goal_achieved=False,
        latest_full_scope_review=str(OUT / 'review.json'), latest_actual_full_human_token_suffix_curriculum=str(OUT / 'complete.json'),
        current_goal_turn_classification=classification)
    atomic(audit_path, audit)
    entry = (f'Full human-code suffix curriculum, **{at}**: all3721original human sources/all14884views independently '
        'reconstruct, preserving every original CE/EOS target.18source tests pass. Full18605reference/prefix controls and '
        'a separate fresh full reader are live; trivial zero-work prefixes and all failures remain explicit. '
        '**No new student training or quality gain is claimed.** The second coding endpoint reports30/48syntax but1/48 '
        'functional; remaining fresh program checks and the final original arm continue. This adapts established '
        'reverse-curriculum research. **All ten final requirements remain incomplete.** '
        '[Evidence](reports/full-human-token-suffix-curriculum-v1-review/metrics.md), '
        '[full review](reports/full-human-token-suffix-curriculum-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name
        path.write_text(entry + path.read_text())
    print(json.dumps(dict(review=str(OUT / 'review.json'), original_human_sources=3721,
        independently_reconstructed_views=14884, actual_new_source_tests=18, new_model_updates=0, goal_achieved=False)))


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19:
        raise ValueError('CUDA-hidden nice19 source-only review required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    main()
