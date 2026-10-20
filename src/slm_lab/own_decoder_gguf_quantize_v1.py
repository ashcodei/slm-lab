"""Explicit Q8_0 derived precision copy of our verified FP32 text component."""
import json
from pathlib import Path

import numpy as np

from .own_decoder_gguf_v1 import file_sha


def quantize_text_component(source, output):
    """Quantize matrices, keep RMSNorm FP32, preserve source and tokenizer IDs."""
    import gguf

    source, output = Path(source), Path(output)
    sidecar = output.with_suffix('.manifest.json')
    source_sidecar = source.with_suffix('.manifest.json')
    meta = json.loads(source_sidecar.read_text())
    if (output.exists() or sidecar.exists() or meta.get('format') != 'own-policy-text-decoder-GGUF-FP32-component-v1'
            or not meta.get('text_component_only_not_full_multimodal_delivery')
            or not meta['original_complete_model_counts']['actual_unique'] < 10**9
            or file_sha(source) != meta['output_sha256']):
        raise ValueError('Fresh destination and exact own FP32 text component required')
    source_manifest_sha = file_sha(source_sidecar)
    reader = gguf.GGUFReader(source)
    writer = gguf.GGUFWriter(output, 'llama')
    rows = []
    try:
        for name, field in reader.fields.items():
            if name.startswith('GGUF.') or name in {'general.architecture', 'general.file_type'}:
                continue
            writer.add_key_value(name, field.contents(), field.types[0],
                sub_type=field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
        writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_Q8_0))
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
        for tensor in reader.tensors:
            if tensor.tensor_type != gguf.GGMLQuantizationType.F32 or not np.isfinite(tensor.data).all():
                raise ValueError('Every source language tensor must be original FP32')
            matrix = tensor.data.ndim == 2
            if matrix and tensor.data.shape[-1] % 32:
                raise ValueError('Every matrix must retain complete supported32-value quantization blocks')
            kind = gguf.GGMLQuantizationType.Q8_0 if matrix else gguf.GGMLQuantizationType.F32
            packed = gguf.quantize(tensor.data, kind) if matrix else tensor.data
            restored = gguf.dequantize(packed, kind) if matrix else packed
            error = float(np.abs(restored - tensor.data).max())
            writer.add_tensor(tensor.name, packed, raw_dtype=kind)
            rows.append(dict(name=tensor.name, original_shape=list(tensor.data.shape),
                original_values=int(tensor.data.size), stored_bytes=int(packed.nbytes),
                precision='Q8_0' if matrix else 'F32', maximum_abs_error=error))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    if file_sha(source) != meta['output_sha256'] or file_sha(source_sidecar) != source_manifest_sha:
        raise ValueError('Original FP32 component and manifest must remain byte-exact')
    if sum(row['original_values'] for row in rows) != meta['exported_language_values']:
        raise ValueError('Every original language tensor/value must remain represented')
    result = dict(format='own-policy-text-decoder-GGUF-Q8_0-component-v1',
        source_FP32_sha256=meta['output_sha256'], source_manifest_sha256=source_manifest_sha,
        source_model_provenance=meta['sources'], original_complete_model_counts=meta['original_complete_model_counts'],
        tokenizer_sha256=meta['tokenizer_sha256'], original_language_values=meta['exported_language_values'],
        tensor_precision_and_error=rows, derived_weight_data_bytes=sum(row['stored_bytes'] for row in rows),
        output_sha256=file_sha(output), quantizer_sha256=file_sha(__file__),
        additional_learned_parameters=0, original_FP32_component_unchanged=True,
        Q8_changes_values_and_can_change_generated_actions=True,
        text_component_only_not_full_multimodal_delivery=True, trained_reward_feature_basis_not_certified=True,
        full_native_quality_not_established_by_quantization=True, goal_achieved=False)
    with sidecar.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
