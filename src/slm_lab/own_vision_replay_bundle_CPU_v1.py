"""Self-contained native own actor bundle after explicit vision/word updates.

Every counted base/head/actor/critic value is present; the old foundation
weights, optimizer and references are not required for inference. This is an
experimental CPU bundle, not a complete broad or human-alignment checkpoint.
"""
import torch

from .corpus import file_hash
from .deterministic_positions_v1 import install, PROTOCOL
from .model import ModelConfig
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_growth_v1 import OwnDenseAssistant, new_backbone

FORMAT = 'own-native-full-base-actor-vision-word-replay-CPU-experiment-v1'


def export(actor, *, sources, completed, full_size_student):
    if type(actor) is not OwnDenseActor or type(completed) is not int or completed < 1 or type(full_size_student) is not bool:
        raise ValueError('Actual complete own actor and explicit completed experimental cursor required')
    actor.assert_shared()
    if full_size_student and actor.counts()['actual_unique'] != 999706442:
        raise ValueError('Actual current full-size student must count every999,706,442 neural value')
    return dict(format=FORMAT, position_interpolation=PROTOCOL, complete_base=actor.base.export(sources),
        complete_actor=actor.export(sources), actor_settings=actor.settings, actual_counts=actor.counts(),
        completed_experimental_updates=completed, actual_full_size_student=full_size_student,
        original_base_weights_or_training_reference_required_for_inference=False,
        complete_broad_source_epoch_or_human_alignment=False, optimizer_or_training_cache_included=False,
        candidate_promoted=False, goal_achieved=False)


def load(path, *, expected_checkpoint_sha256, expected_sources, seed=970817):
    if file_hash(path) != expected_checkpoint_sha256: raise ValueError('Exact own native checkpoint SHA required')
    raw = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (raw.get('format') != FORMAT or raw.get('position_interpolation') != PROTOCOL
            or type(raw.get('completed_experimental_updates')) is not int or raw['completed_experimental_updates'] < 1
            or type(raw.get('actual_full_size_student')) is not bool
            or raw.get('original_base_weights_or_training_reference_required_for_inference') is not False
            or raw.get('complete_broad_source_epoch_or_human_alignment') is not False
            or raw.get('optimizer_or_training_cache_included') is not False
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False
            or raw['complete_base']['sources'] != expected_sources or raw['complete_actor']['sources'] != expected_sources):
        raise ValueError('Exact complete own CPU experimental native bundle identity and sources required')
    base = install(OwnDenseAssistant(new_backbone(ModelConfig(**raw['complete_base']['config']), seed), **raw['complete_base']['settings'])).eval()
    base.import_export(raw['complete_base'], expected_sources=expected_sources)
    actor = OwnDenseActor(base, **raw['actor_settings']).eval()
    actor.import_export(raw['complete_actor'], expected_sources=expected_sources)
    if (actor.counts() != raw['actual_counts'] or actor.counts()['actual_unique'] >= 10**9
            or raw['actual_full_size_student'] and actor.counts()['actual_unique'] != 999706442
            or file_hash(path) != expected_checkpoint_sha256):
        raise ValueError('Every counted native student value and immutable source must agree')
    return actor, dict(actual_counts=actor.counts(), actual_updates=raw['completed_experimental_updates'],
        actual_full_size_student=raw['actual_full_size_student'], complete_broad_or_human_alignment=False,
        self_contained_own_weights_no_old_base_or_teacher_loaded=True)
