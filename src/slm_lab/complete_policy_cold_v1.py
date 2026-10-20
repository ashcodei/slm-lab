"""Cold-check the complete policy and original full16-word source batch."""
import torch

from .codenet_checkpoint_v1 import capture_rng
from .own_dense_actor_v1 import OwnDenseActor
from .parallel_function_repair_stage_v1 import value_digest
from .whole_human_sft_evidence_v1 import tensor_digest


def _cold(stage, independent_actor, saved, inputs):
    error = None
    other = report = None
    before_rng = value_digest(capture_rng(stage.device))
    try:
        if (not callable(independent_actor) or set(inputs) != {'text', 'photo', 'word'}
                or set(inputs['text']) != {'input_ids', 'attention_mask'}
                or set(inputs['photo']) != {'input_ids', 'attention_mask', 'pixel_values'}
                or set(inputs['word']) != {'word_pixels'}):
            raise ValueError('Independent own actor and complete text/photo/word inputs are required')
        other = independent_actor()
        if (type(other) is not OwnDenseActor or other.counts() != stage.model.counts()
                or other.settings != stage.model.settings or other.config != stage.model.config
                or any(p.device != stage.device or p.dtype != torch.float32 or p.grad is not None for p in other.parameters())
                or {p.data_ptr() for p in other.parameters()} & {p.data_ptr() for p in stage.model.parameters()}
                or value_digest(capture_rng(stage.device)) != before_rng):
            raise ValueError('A separately loaded identical own base/actor graph must preserve all private RNGs and share no storage')
        for name in ['text', 'photo']:
            ids, mask = inputs[name]['input_ids'], inputs[name]['attention_mask']
            if (ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.long or mask.dtype != torch.bool
                    or ids.shape != mask.shape or not bool(mask.all()) or ids[0, 0] != 1
                    or not bool(((0 <= ids) & (ids < other.config.vocab_size)).all())):
                raise ValueError('Every actual complete cold native text position/mask must be retained')
        photo, word = inputs['photo']['pixel_values'], inputs['word']['word_pixels']
        if (photo.ndim != 4 or photo.shape[:2] != (1, 3) or photo.shape[2] != photo.shape[3]
                or word.ndim != 4 or word.shape[0] < 1 or word.shape[1:3] != (3, 64)
                or any(t.dtype != torch.float32 or not bool(torch.isfinite(t).all()) for t in [photo, word])
                or stage.mode == 'capacity' and (inputs['text']['input_ids'].shape != (1, 4096)
                    or photo.shape != (1, 3, 448, 448) or word.shape != (16, 3, 64, 448))):
            raise ValueError('Actual full-size capacity retains4096text/448RGB/64x448whole-word inputs')
        other.import_export(saved['model'], expected_sources=stage.sources)
        hot = dict(stage.model.named_parameters())
        if (list(hot) != [n for n, _ in other.named_parameters()]
                or any(not torch.equal(hot[n], p) for n, p in other.named_parameters())):
            raise ValueError('Every original base/head/policy/critic value must exactly match independent cold recovery')
        stage.model.eval()
        other.eval()
        batches = {n: {k: v.to(stage.device) for k, v in b.items()} for n, b in inputs.items()}
        output_digests = {}
        with torch.no_grad(), torch.autocast(stage.device.type, enabled=False):
            for name in ['text', 'photo']:
                wanted = stage.model(**batches[name])['logits']
                actual = other(**batches[name])['logits']
                if (wanted.dtype != torch.float32 or not bool(torch.isfinite(wanted).all())
                        or not torch.equal(wanted, actual)):
                    raise ValueError('Every complete FP32text/photo actor output must cold-load bit exactly')
                output_digests[name] = tensor_digest([('logits', actual)])
                del wanted, actual
            wanted = stage.model.base.word_logits(batches['word']['word_pixels'])
            actual = other.base.word_logits(batches['word']['word_pixels'])
            if wanted.dtype != torch.float32 or not bool(torch.isfinite(wanted).all()) or not torch.equal(wanted, actual):
                raise ValueError('The complete original FP32whole-word readout must remain cold-load bit exact')
            output_digests['word'] = tensor_digest([('logits', actual)])
            actual_word_shape = list(actual.shape)
            if stage.mode == 'capacity' and actual_word_shape != [16, 140, 257]:
                raise ValueError('Every original16word/140frame/257byte-logit output must remain complete')
        if value_digest(capture_rng(stage.device)) != before_rng:
            raise ValueError('All actual independent cold loads and complete forward probes must preserve private RNGs')
        report = dict(complete=True, mode=stage.mode, mechanics_only=stage.mode == 'mechanics',
            actual_text_positions=inputs['text']['input_ids'].shape[1], actual_photo_side=photo.shape[-1],
            actual_word_shape=list(word.shape[-2:]), actual_word_readout_dtype='torch.float32',
            actual_word_batch_size=word.shape[0], actual_word_readout_shape=actual_word_shape,
            every_base_head_actor_critic_parameter_equal=True, no_shared_parameter_storage=True,
            complete_text_photo_word_outputs_bit_exact=True, all_private_RNGs_unchanged=True,
            actual_complete_output_sha256=output_digests, original_own_base_plus_committed_actor_parts_loaded=True,
            final_standalone_export_and_final_inference_still_required=True)
    except Exception as exc:
        error = repr(exc)
    stage.coordinated(error, 'independent complete actor cold')
    del other
    return report

