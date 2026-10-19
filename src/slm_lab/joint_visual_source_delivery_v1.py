"""Deliver one complete original visual schedule entry without a neural model.

The adapter keeps its independent source/pixel admission. This layer supplies
the global objective metadata alongside each actual rank's tensor payload;
it does not invent negatives, truncate words, or admit a GPU job.
"""
import hashlib
from pathlib import Path

from .corrected_multimodal_batches_v1 import CorrectedMultimodalBatches, sha, state
from .joint_visual_objective_update_v1 import FAMILIES


class OriginalVisualSources:
    def __init__(self, adapter):
        if type(adapter) is not CorrectedMultimodalBatches:
            raise ValueError('The exact original complete corrected source adapter is required')
        self.adapter = adapter
        self.purpose = adapter.purpose
        self.training_admitted = adapter.training_admitted
        self.admission_path = adapter.folder / 'source-training-admission.json'
        self.admission_state = state(self.admission_path) if self.training_admitted else None
        self.bindings = dict(corrected_index_sha256=sha(adapter.index_path),
            tokenizer_sha256=sha(adapter.root / 'data/tokenizer-corpus/tokenizer.json'),
            complete_schedule_sha256=sha(adapter.folder / 'complete-corrected-single-epoch-schedule.json'),
            source_admission_sha256=sha(self.admission_path) if self.training_admitted else None)

    def prepare(self, entry, *, rank, world, device):
        if (type(rank) is not int or type(world) is not int or not 0 <= rank < world <= 4
                or entry['proposed_rank_source_counts'] !=
                [len(entry['original_local_positions'][r::world]) for r in range(world)]):
            raise ValueError('Every actual rank must retain its original source assignment')
        if self.training_admitted and state(self.admission_path) != self.admission_state:
            raise ValueError('The independently admitted complete source receipt changed')
        data = self.adapter
        family, positions = entry['family'], entry['original_local_positions']
        if family not in FAMILIES or not positions:
            raise ValueError('One nonempty original homogeneous global entry is required')
        lengths = [data.lengths[family][p] for p in positions]
        targets = [data.targets[family][p] for p in positions]
        if (sum(lengths) != entry['complete_context_tokens']
                or sum(targets) != entry['response_EOS_targets']):
            raise ValueError('The complete current index budgets must equal the bound schedule entry')
        source = dict(family=family, global_positions=list(positions),
            global_context_lengths=lengths, global_response_EOS_targets=targets)
        kind = FAMILIES[family]
        if kind in ['word', 'photo']:
            rows = [data.row(family, p) for p in positions]
            texts = [row['current_row']['response'] for row in rows]
            if kind == 'word':
                source.update(global_word_UTF8_bytes=[len(t.encode('utf8')) for t in texts],
                    global_word_UTF8_sha256=[hashlib.sha256(t.encode('utf8')).hexdigest() for t in texts])
            else:
                keys = []
                for row in rows:
                    metadata = row['metadata']
                    inventory = data.inventory.get(str(Path(metadata['image_path']).resolve()), {})
                    key = metadata['original_RGB448_metadata_sha256'] or inventory.get('native_RGB448')
                    if not isinstance(key, str) or len(key) != 64:
                        raise ValueError('Each original complete caption needs its exact native RGB448 identity')
                    keys.append(key)
                source.update(pixel_keys=keys, captions=texts)
        local = positions[rank::world]
        delivered = data.batch(family, local, device=device)
        if not local:
            if delivered is not None:
                raise ValueError('An empty rank must not duplicate any original source')
            return None, source
        if (delivered['source_role'] != 'train' or delivered['purpose'] != self.purpose
                or delivered['training_admitted'] is not self.training_admitted
                or delivered['complete_context_tokens'] != sum(lengths[rank::world])
                or delivered['response_EOS_targets'] != sum(targets[rank::world])
                or [r['original_local_position'] for r in delivered['source_observations']] != local
                or any(r['family'] != family for r in delivered['source_observations'])):
            raise ValueError('Every original local occurrence and complete source budget must reach delivery')
        return {key: delivered[key] for key in ['native', 'word', 'matching']}, source
