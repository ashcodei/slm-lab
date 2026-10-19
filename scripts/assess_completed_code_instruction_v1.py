"""Assess the completed own full-model coding stage with unchanged fixed protocols."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import time

import torch

import assess_codenet_model_v2 as code_assessment
from assess_code_residual_lr_pilot_v1 import compare_fixed_results
from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from slm_lab import knowledge_evaluation
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from train_multimodal_pilot import complete_stage

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA = '9e5166b7de225a7ef691111fec83ff25f5f1cbd067682037f591b565b0d78483'


def main():
    reports = ROOT / 'reports/completed-code-instruction-assessment-v1'
    if reports.exists():
        raise ValueError('Preserve prior complete or partial assessment evidence')
    checkpoint = ROOT / 'checkpoints/extended-curriculum-v1/multilingual-code-instruction/model.pt'
    plan_path = ROOT / 'reports/extended-curriculum-v1/plan.json'
    plan = json.loads(plan_path.read_text())
    spec = next(s for s in plan['stages'] if s['name'] == 'multilingual-code-instruction')
    if (spec['steps'] != 28343 or not complete_stage(checkpoint.parent, spec)
            or file_hash(checkpoint) != CHECKPOINT_SHA):
        raise ValueError('Exact completed own coding checkpoint and frozen stage budget required')
    paths = [Path(__file__), checkpoint, checkpoint.parent / 'training.json',
        checkpoint.parent / 'tokenizer.json', plan_path,
        ROOT / 'scripts/assess_codenet_model_v2.py',
        ROOT / 'scripts/assess_code_residual_lr_pilot_v1.py',
        ROOT / 'scripts/prepare_codenet_native_contract_pack_v2.py',
        ROOT / 'scripts/train_multimodal_pilot.py',
        ROOT / 'src/slm_lab/knowledge_evaluation.py',
        ROOT / 'src/slm_lab/codenet_checkpoint_v1.py',
        ROOT / 'data/knowledge-eval-v1/questions.jsonl',
        ROOT / 'data/knowledge-eval-v1/questions.audit.json']
    bound = {str(p): file_hash(p) for p in paths}
    guard = FileBindings(bound)
    reports.mkdir()

    def status(message, **fields):
        atomic_json(reports / 'status.json', dict(status=message,
            updated_at=datetime.now(timezone.utc).isoformat(), optimizer_updates=0,
            candidate_promoted=False, goal_achieved=False, **fields))

    atomic_json(reports / 'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(checkpoint), checkpoint_sha256=CHECKPOINT_SHA, completed_updates=28343,
        planned_updates=28343, model_parameters=706012672, source_files_sha256=bound,
        code_questions=28, code_generation_budget=2048, raw_QA_questions=559,
        raw_QA_generation_budget=32, device='cuda:0', maximum_GPU_memory_fraction=.14,
        primary_generation_parsing_scoring_or_architecture_loading_changed=False,
        complete_generation_sealed_before_execution=True, optimizer_updates=0,
        candidate_promoted=False, goal_achieved=False,
        scope='Completed original code/instruction stage, with later math and image stages still pending. Fixed development assessments retain prior-exposure qualifications; no certified independent final or frontier-model comparison.'))
    try:
        # A lack of capacity is a pending condition, never permission to lower the
        # existing evaluator's writer reservation or start another assessment.
        torch.set_num_threads(1)
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise ValueError('Original BF16 inference protocol requires compatible CUDA')
        while True:
            guard.check()
            occupied = occupied_writer_reservation()
            required = 26 * 1024**3 - occupied['credited_bytes'] + 128 * 1024**2
            free = shutil.disk_usage(ROOT).free
            gpu_free = torch.cuda.mem_get_info(device)[0]
            if free >= required and gpu_free >= 12 * 1024**3:
                break
            status('waiting for unchanged evaluator storage and shared GPU capacity',
                free_home_bytes=free, required_home_bytes=required,
                free_GPU_bytes=gpu_free, minimum_initial_free_GPU_bytes=12 * 1024**3)
            time.sleep(15)
        code_folder = ROOT / 'reports/completed-code-instruction-codenet-v1'
        status('generating every fixed raw code answer from completed full-model coding checkpoint')
        code_assessment.evaluate(checkpoint, code_folder, 'cuda:0', threads=1, memory_fraction=.14,
            expected_metadata=dict(stage='sft', steps=28343, planned_steps=28343,
                world_size=4, data_sha256=spec['data_sha256']),
            label='own706M_completed_original_code_instruction_stage_28343_updates',
            expected_checkpoint_sha256=CHECKPOINT_SHA)
        guard.check(full=True)
        knowledge_path = reports / 'raw-knowledge.json'
        status('assessing every fixed raw broad-domain question from completed coding checkpoint')
        knowledge_evaluation.evaluate(checkpoint, ROOT / 'data/knowledge-eval-v1/questions.jsonl',
            knowledge_path, 'cuda:0', max_examples=559, max_new_tokens=32)
        guard.check(full=True)
        # Reuse the already-tested complete input/gold/score verification, then
        # supply the correct full-model identity instead of its pilot schema.
        with tempfile.TemporaryDirectory(prefix='comparison-validation-', dir=reports) as scratch:
            comparison = compare_fixed_results(code_folder, knowledge_path, Path(scratch) / 'readback.json')
        comparison['candidate_raw_knowledge'] = comparison.pop('candidate_code_route_raw_knowledge')
        comparison['candidate_identity'] = dict(checkpoint=str(checkpoint),
            checkpoint_sha256=CHECKPOINT_SHA, parameters=706012672, steps=28343,
            planned_steps=28343, model_kind='original full model; no residual route')
        comparison['scope'] = ('Own706M completed 28343-update original coding stage versus its own '
            'completed-language checkpoint, using all unchanged28 code questions/2048 raw tokens '
            'and559 raw QA questions/32 tokens. Later curriculum stages remain pending. These '
            'are reused development assessments with earlier-exposure qualifications, not '
            'certified independent final or frontier-model evidence.')
        atomic_json(reports / 'matched-comparison.json', comparison)
        status('fixed completed-coding-stage assessment complete; capability evidence available',
            code=comparison['candidate_code'], knowledge=comparison['candidate_raw_knowledge'])
        print('completed_coding_stage_fixed_assessment_complete', flush=True)
    except Exception as error:
        status('assessment stopped; inspect preserved evidence before continuation', error=str(error))
        raise


if __name__ == '__main__':
    main()
