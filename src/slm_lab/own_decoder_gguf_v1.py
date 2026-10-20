"""Export our CPU policy's text decoder as an explicit FP32 GGUF component.

The compatible architecture tag selects runtime operators, not outside weights.
Vision, word, reward and critic heads are not contained in this text component.
Folding adapters changes floating-point operation order and is not certified to
preserve the feature basis of a trained reward head.
"""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch

from .own_dense_actor_v1 import OwnDenseActor
from .own_code_delta_v1 import matrix_names
from .whole_human_sft_evidence_v1 import tensor_digest


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@torch.no_grad()
def export_text_component(model, tokenizer_path, output, *, sources):
    """Write a fresh text-only component; leave every source value/flag exact."""
    import gguf

    output = Path(output)
    sidecar = output.with_suffix('.manifest.json')
    tokenizer_path = Path(tokenizer_path)
    if (type(model) is not OwnDenseActor or model.training or output.exists() or sidecar.exists()
            or torch.is_autocast_enabled('cpu')
            or any(p.device.type != 'cpu' or p.dtype != torch.float32 or p.grad is not None for p in model.parameters())
            or not sources or any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64
                or any(c not in '0123456789abcdef' for c in v) for k, v in sources.items())):
        raise ValueError('Fresh output, exact own eval-mode CPU FP32 policy and explicit source SHAs required')
    model.assert_shared()
    config = model.config
    raw = json.loads(tokenizer_path.read_text())
    pre = raw.get('pre_tokenizer') or {}
    if (raw['model']['type'] != 'BPE' or raw.get('normalizer') is not None
            or pre.get('type') != 'ByteLevel' or pre.get('add_prefix_space') is not False
            or pre.get('use_regex') is not True):
        raise ValueError('Exact original unnormalized ByteLevel BPE tokenizer required')
    vocabulary = raw['model']['vocab']
    if len(vocabulary) != config.vocab_size or sorted(vocabulary.values()) != list(range(config.vocab_size)):
        raise ValueError('Every original vocabulary ID must remain dense and unchanged')
    tokens = [''] * config.vocab_size
    for text, index in vocabulary.items():
        tokens[index] = text
    if tokens[:4] != ['<pad>', '<bos>', '<eos>', '<unk>']:
        raise ValueError('Original pad/bos/eos/unk IDs 0/1/2/3 required')
    merges = [value if isinstance(value, str) else ' '.join(value) for value in raw['model']['merges']]
    before = tensor_digest(model.named_parameters())
    signature = [(id(p), p._version, p.data_ptr(), p.requires_grad) for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    tokenizer_sha = file_sha(tokenizer_path)
    writer = gguf.GGUFWriter(output, 'llama')
    count = 0
    tensor_names = []

    def add(name, tensor):
        nonlocal count
        if tensor.device.type != 'cpu' or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise ValueError('Every derived language tensor must be finite original-precision CPU data')
        writer.add_tensor(name, tensor.detach().contiguous().numpy())
        count += tensor.numel()
        tensor_names.append(name)

    try:
        writer.add_name('Own SLM text decoder component')
        writer.add_file_type(int(gguf.LlamaFileType.ALL_F32))
        writer.add_context_length(config.max_text_length)
        writer.add_embedding_length(config.hidden_size)
        writer.add_block_count(config.num_layers)
        writer.add_feed_forward_length(config.intermediate_size)
        writer.add_head_count(config.num_heads)
        writer.add_head_count_kv(config.num_kv_heads)
        writer.add_rope_dimension_count(config.hidden_size // config.num_heads)
        writer.add_rope_freq_base(config.rope_theta)
        writer.add_layer_norm_rms_eps(config.norm_eps)
        writer.add_tokenizer_model('gpt2')
        writer.add_tokenizer_pre('gpt-2')
        writer.add_token_list(tokens)
        writer.add_token_scores([0.] * len(tokens))
        writer.add_token_types([*([gguf.TokenType.CONTROL] * 3), gguf.TokenType.UNKNOWN,
            *([gguf.TokenType.NORMAL] * (len(tokens) - 4))])
        writer.add_token_merges(merges)
        writer.add_unk_token_id(3)
        writer.add_bos_token_id(1)
        writer.add_eos_token_id(2)
        writer.add_pad_token_id(0)
        writer.add_add_bos_token(False)
        writer.add_add_eos_token(False)
        network = model.base.network
        add('token_embd.weight', network.embedding.weight)
        add('output_norm.weight', model.parts['norms'][-1].weight)
        labels = ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down')
        originals = matrix_names(config.num_layers)
        for i in range(config.num_layers):
            for offset, label in enumerate(('attn_norm', 'ffn_norm')):
                add(f'blk.{i}.{label}.weight', model.parts['norms'][2 * i + offset].weight)
            for j, label in enumerate(labels):
                position = 7 * i + j
                original = network.get_parameter(originals[position])
                parts = model.parts['linears'][position]
                add(f'blk.{i}.{label}.weight', original.detach() + parts.up.detach() @ parts.down.detach())
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    if (before != tensor_digest(model.named_parameters()) or signature !=
            [(id(p), p._version, p.data_ptr(), p.requires_grad) for p in model.parameters()]
            or not torch.equal(rng, torch.get_rng_state()) or file_sha(tokenizer_path) != tokenizer_sha):
        raise ValueError('Every original value, identity, version, flag, RNG and tokenizer byte must remain unchanged')
    manifest = dict(format='own-policy-text-decoder-GGUF-FP32-component-v1', config=asdict(config),
        sources=dict(sources), original_complete_model_counts=model.counts(), exported_language_values=count,
        tensor_names=tensor_names, exported_language_FP32_bytes=4 * count,
        output_sha256=file_sha(output), original_full_parameter_sha256=before, tokenizer_sha256=tokenizer_sha,
        exporter_sha256=file_sha(__file__), gguf_package=importlib.metadata.version('gguf'),
        original_model_and_CPU_RNG_unchanged=True, additional_learned_parameters=0,
        original_interleaved_RoPE_rows_not_HF_permuted=True, tied_output_reuses_original_vocabulary_table=True,
        adapter_folding_changes_float_operation_order=True, trained_reward_feature_basis_not_certified=True,
        text_component_only_not_full_multimodal_delivery=True, neural_backend_parity_not_proven_by_export=True,
        native_training_and_assessments_not_changed=True, goal_achieved=False)
    with sidecar.open('x') as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest
