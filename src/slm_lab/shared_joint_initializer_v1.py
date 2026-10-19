"""One own native initializer from fixed completed text and visual sources.

Keep the complete text source's language, critic and remaining heads. Copy
only the compatible visual encoder, word reader and matching head. The result
has independent storage and no added values. Image-answer quality is not
preserved automatically when the decoder and visual features are combined.
Completed-source admission and all capacity/quality checks remain external.
"""
import copy
from dataclasses import asdict

import torch

from .codenet_checkpoint_v1 import capture_rng
from .own_dense_growth_v1 import OwnDenseAssistant
from .parallel_function_repair_stage_v1 import value_digest
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-single-native-joint-text-and-visual-initializer-v1'
VISUAL_PREFIXES = ('network.vision.', 'heads.word.', 'heads.pointing.')


@torch.no_grad()
def assemble(text, visual, *, sources, working_budget_bytes, allow_mechanics=False):
    if (type(text) is not OwnDenseAssistant or type(visual) is not OwnDenseAssistant or text is visual
            or asdict(text.config) != asdict(visual.config) or text.settings != visual.settings
            or text.training or visual.training or text.counts() != visual.counts()
            or not sources or any(type(k) is not str or not k or type(v) is not str or len(v) != 64
                or any(c not in '0123456789abcdef' for c in v) for k, v in sources.items())):
        raise ValueError('Two distinct compatible own eval-mode native sources and exact source identities are required')
    count = text.counts()['total_unique']
    full = count == 984399690 and text.config.num_layers == 35 and text.heads['word'].expansion == 5
    if (not full and not (allow_mechanics is True and count < 10**7)
            or any(p.device.type != 'cpu' or p.dtype != torch.float32 or p.grad is not None or not bool(torch.isfinite(p).all())
                for model in [text, visual] for p in model.parameters())):
        raise ValueError('Full-size own CPU FP32 natives or an explicit small mechanics control are required')
    needed = 4 * count + max(p.numel() for p in text.parameters()) + 128 * 1024**2
    if type(working_budget_bytes) is not int or working_budget_bytes < needed:
        raise ValueError('The complete independent native clone and finite-check working memory must be funded')
    before = dict(text=tensor_digest(text.named_parameters()), visual=tensor_digest(visual.named_parameters()),
        RNG=value_digest(capture_rng('cpu')))
    result = copy.deepcopy(text).eval(); original = dict(visual.named_parameters())
    copied = []
    for name, parameter in result.named_parameters():
        if name.startswith(VISUAL_PREFIXES):
            parameter.copy_(original[name]); copied.append(name)
    result.assert_inventory()
    wanted = {n: p for n, p in text.named_parameters()}
    wanted.update({n: p for n, p in visual.named_parameters() if n.startswith(VISUAL_PREFIXES)})
    aliases = {p.data_ptr() for model in [text, visual] for p in model.parameters()}
    if (not copied or result.counts() != text.counts() or result.network.embedding.weight is not result.network.lm_head.weight
            or any(p.data_ptr() in aliases or not torch.equal(p, wanted[n]) for n, p in result.named_parameters())
            or before != dict(text=tensor_digest(text.named_parameters()), visual=tensor_digest(visual.named_parameters()),
                RNG=value_digest(capture_rng('cpu')))):
        raise ValueError('Every chosen original value, independent storage, tied vocabulary and source/RNG must remain exact')
    return result, dict(format=FORMAT, sources=copy.deepcopy(sources), actual_counts=result.counts(),
        copied_visual_word_matching_names=copied, copied_visual_word_matching_values=sum(wanted[n].numel() for n in copied),
        actual_text_initializer_sha256=tensor_digest(result.language_named_parameters()),
        actual_visual_initializer_sha256=tensor_digest((n, p) for n, p in result.named_parameters() if n.startswith(VISUAL_PREFIXES)),
        original_native_text_critic_and_other_heads_preserved=True, every_selected_source_value_exact=True,
        source_parameters_and_all_private_CPU_RNG_unchanged=True, independent_result_storage=True,
        required_additional_working_bytes=needed, declared_additional_working_bytes=working_budget_bytes,
        original_source_completion_not_proved_by_this_value_assembly=True,
        actual_full_size_value_assembly=full, image_answer_quality_and_GPU_capacity_not_proved=True,
        new_neural_values=0, optimizer_updates=0, candidate_promoted=False, goal_achieved=False)
