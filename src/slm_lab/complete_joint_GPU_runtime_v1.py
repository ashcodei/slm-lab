"""Connect the fixed own native, complete original data and joint GPU stages.

The owning parent supplies its actual lease/funding guard and immutable plan.
The same CPU factory independently rebuilds capacity/final native exports;
it never imports selected weights from an exported candidate into the source.
"""
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .complete_joint_capacity_stage_v2 import CompleteJointCapacityStageV2
from .complete_joint_indexed_training_v4 import CompleteJointIndexedTrainingV4
from .complete_joint_objective_bridge_v3 import complete_entries
from .complete_joint_owned_recipe_v1 import FORMAT, FULL, OPTIONS, OPTIMIZER, ACTOR
from .complete_joint_training_data_v1 import CompleteJointTrainingDataV1
from .corpus import file_hash
from .frozen_own_text_reference_v1 import FrozenOwnTextReferenceV1
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_standalone_v1 import load
from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_actor_vision_v1 import SharedJointActorVisionV1
from .shared_joint_checkpoint_v2 import FixedGuard
from .whole_human_sft_evidence_v1 import tensor_digest


def build(root, plan_path, *, mode, device='cpu'):
    import json
    root, plan_path = Path(root).resolve(), Path(plan_path).resolve(); plan = json.loads(plan_path.read_text())
    if (mode not in ['capacity', 'train'] or plan.get('format') != FORMAT or plan.get('full_original_budget') != FULL
            or plan.get('options') != OPTIONS or plan.get('optimizer') != OPTIMIZER or plan.get('actor_settings') != ACTOR
            or plan.get('expected_native_values') != 984399690 or plan.get('expected_research_values') != 999706442):
        raise ValueError('The exact owned full original joint model,source/options and complete optimization recipe is required')
    files = dict(plan['source_files_sha256']); files[str(plan_path)] = file_hash(plan_path)
    guard = FileBindings(files); spec = plan['initializer']
    native, metadata = load(spec['checkpoint'], expected_checkpoint_sha256=spec['checkpoint_sha256'], expected_sources=spec['sources'])
    if (native.counts()['total_unique'] != 984399690 or metadata['completed_updates'] != 0
            or metadata['provenance'].get('kind') != 'own-completed-human-DPO-and-own-foundation-joint-initializer-v1'
            or metadata['tokenizer_sha256'] != spec['tokenizer_sha256']):
        raise ValueError('The exact complete own CPU initializer and original native vocabulary must precede every fresh stage')
    view = SharedJointActorVisionV1(OwnDenseActor(native, **ACTOR)).eval().to(device)
    if (view.actor.counts()['actual_unique'] != 999706442 or len(view.named_selected()) != 690
            or sum(p.numel() for _, p in view.named_selected()) != 68123713):
        raise ValueError('All original counted frozen and selected native/actor/vision/word values must remain')
    source = CompleteJointTrainingDataV1(root, view.actor.config, word_expansion=5)
    capacity = json.loads(Path(plan['capacity_plan']).read_text())
    if (file_hash(plan['capacity_plan']) != plan['capacity_plan_sha256'] or capacity['source_identity_sha256'] != source.identity_sha256
            or capacity['actual_complete_capacity_cases'] != 47 or capacity['actual_original_joint_steps'] != 5362):
        raise ValueError('The full original actual-source capacity recipe and complete data identity must agree')
    entries = [r['checkpoint_entry'] for r in capacity['cases']] if mode == 'capacity' else complete_entries(source)
    optimizer = torch.optim.AdamW([p for _, p in view.named_selected()], **dict(OPTIMIZER, betas=tuple(OPTIMIZER['betas'])))
    fixed = FixedGuard(view, optimizer)
    reference = FrozenOwnTextReferenceV1(view.actor.base,
        sources={'fixed_own_initializer_sha256': spec['checkpoint_sha256']}, tokenizer_sha256=source.tokenizer_sha256)
    sources = dict(text_initializer_sha256=tensor_digest(view.actor.base.language_named_parameters()),
        visual_initializer_sha256=tensor_digest((n, p) for n, p in view.actor.base.named_parameters()
            if n.startswith(('network.vision.', 'heads.pointing.', 'heads.word.'))),
        tokenizer_sha256=source.tokenizer_sha256, run_plan_sha256=file_hash(plan_path), source_admission_sha256=source.identity_sha256,
        complete_schedule_sha256=value_digest(entries), fixed_values_sha256=fixed.sha256)
    stage_files = dict(source.files)
    for p, h in files.items():
        if p in stage_files and stage_files[p] != h: raise ValueError('The exact actual source and owned initializer bindings disagree')
        stage_files[p] = h
    return dict(plan=plan, view=view, source=source, capacity=capacity, entries=entries, optimizer=optimizer,
        fixed_guard=fixed, reference=reference, sources=sources, initializer_guard=guard, source_files=stage_files)


def stage(parts, reports, *, mode, control_group, gradient_group):
    reports = Path(reports)
    keywords = dict(history_root=reports / mode / 'history',
        history_budgets=parts['plan']['storage_budget']['capacity_history' if mode == 'capacity' else 'training_history'],
        initializer_guard=parts['initializer_guard'], maximum_attempts=2 if mode == 'capacity' else 1,
        control_group=control_group, gradient_group=gradient_group)
    args = (parts['source'], parts['view'], parts['reference'], parts['optimizer'], parts['sources'], parts['plan']['options'])
    if mode == 'capacity':
        import json
        visual = json.loads((parts['source'].root / 'reports/joint-visual-actual-capacity-cases-CPU-v1/actual-complete-source-capacity-plan.json').read_text())
        return CompleteJointCapacityStageV2(*args, capacity_plan=parts['capacity'], original_visual_capacity=visual, **keywords)
    if mode != 'train': raise ValueError('Only the full original capacity or complete training stage is allowed')
    return CompleteJointIndexedTrainingV4(*args, **keywords)
