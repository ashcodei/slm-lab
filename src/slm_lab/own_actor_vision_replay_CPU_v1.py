"""Caption/matching updates with whole original OCR replay on one own actor.

No new parameters or outside encoder are introduced. Complete original word
CTC and native response/EOS CE train the existing vision/word branches while
all language/actor/critic/other heads stay fixed. This CPU-only mechanism does
not claim universal OCR retention, visual gains, or four-GPU admission.
"""
import math

import torch

from .own_actor_vision_alignment_v1 import OwnActorVisionAlignment
from .shared_word_data_v1 import SharedWordData
from .shared_word_stage_v1 import ctc_word_sum


class OwnActorVisionReplayCPU(OwnActorVisionAlignment):
    def named_selected(self):
        return super().named_selected() + [('word.' + n, p) for n, p in self.actor.base.heads['word'].named_parameters()]

    def backward_with_replay(self, photo_objective, word_data, positions, *, CTC_weight=.25, native_word_CE_weight=.25):
        if (next(self.parameters()).device.type != 'cpu' or any(p.dtype != torch.float32 for p in self.parameters())
                or type(word_data) is not SharedWordData or word_data.role != 'train' or word_data.purpose != 'optimization'
                or not positions or len(set(positions)) != len(positions)
                or any(type(p) is not int or not 0 <= p < len(word_data) for p in positions)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in [CTC_weight, native_word_CE_weight])):
            raise ValueError('Own FP32 CPU model, distinct admitted original train words and explicit positive replay weights required')
        parameters = [p for _, p in self.named_selected()]; chosen = {id(p) for p in parameters}
        fixed = [p for p in self.parameters() if id(p) not in chosen]
        signature = lambda: [(id(p), p._version, p.data_ptr(), p.requires_grad) for p in fixed]
        initial = signature()
        if any(p.grad is not None for p in self.parameters()): raise ValueError('Every gradient must start clear')
        rows = [word_data.rows[p] for p in positions]
        word_targets = sum(r['native']['length'] - r['native']['target_start'] for r in rows)
        if not word_targets or any(r['split'] != 'train' for r in rows): raise ValueError('Every complete original train transcript/EOS target required')
        if not bool(torch.isfinite(photo_objective['loss'])): raise ValueError('Finite native photo/matching objective required')
        photo_objective['loss'].backward(inputs=parameters)
        ctc_total = native_total = byte_targets = 0
        for position in positions:
            batch = word_data.batch([position], 'cpu')
            logits = self.actor.base.word_logits(batch['word_pixels'])
            ctc, byte_count = ctc_word_sum(logits, batch['texts'])
            (ctc * CTC_weight / len(positions)).backward(inputs=parameters)
            ctc_total += float(ctc.detach()); byte_targets += byte_count
            del ctc, logits
            native = {k: batch[k] for k in ['input_ids', 'attention_mask', 'pixel_values', 'labels']}
            targets = int((batch['labels'][:, 1:] != -100).sum())
            if not targets or any(int(row[row != -100][-1]) != 2 for row in batch['labels']):
                raise ValueError('Every full original word response must retain its genuine EOS target')
            native_loss = self.native(**native)['loss']
            if not bool(torch.isfinite(native_loss)): raise ValueError('Complete native word CE must be finite')
            (native_loss * native_word_CE_weight * targets / word_targets).backward(inputs=parameters)
            native_total += float(native_loss.detach()) * targets
            del native_loss
        if (signature() != initial or any(p.grad is not None for p in fixed)
                or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters)):
            raise ValueError('Every selected vision/matching/word gradient must be finite and all fixed parameters untouched')
        return dict(photo_CE=float(photo_objective['caption_CE'].detach()), matching_loss=float(photo_objective['matching_loss'].detach()),
            photo_objective=float(photo_objective['loss'].detach()), photo_response_EOS_targets=photo_objective['response_EOS_targets'],
            original_replayed_word_positions=list(positions), original_complete_words=len(positions), UTF8_byte_targets=byte_targets,
            original_word_response_EOS_targets=word_targets, word_mean_byte_normalized_CTC=ctc_total / len(positions),
            word_native_token_mean_CE=native_total / word_targets, CTC_weight=CTC_weight, native_word_CE_weight=native_word_CE_weight,
            actual_summed_objective=float(photo_objective['loss'].detach()) + CTC_weight * ctc_total / len(positions)
                + native_word_CE_weight * native_total / word_targets,
            all_original_transcripts_bytes_and_EOS_preserved=True, no_extra_neural_values=True,
            no_held_words_or_external_teacher=True, no_global_retention_or_GPU_admission_claim=True)
