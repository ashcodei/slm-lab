"""Explicit longer CTC time axis with the same complete shared-model weights.

Repeated patch features give the recurrent readout more alignment steps. They
do not add pixel detail or establish reading ability. Original v1 evidence and
its artifact format stay unchanged; v2 exports state this different computation.
"""
import torch
from torch import nn

from .own_aligned_assistant_v1 import OwnAlignedAssistant, FORMAT as PRIOR_FORMAT

FORMAT = 'own-single-backbone-alignment-vision-expanded-word-extensions-v2'


class ExpandedByteWordHead(nn.Module):
    def __init__(self, previous, expansion):
        super().__init__()
        if type(expansion) is not int or not 1 <= expansion <= 8:
            raise ValueError('Explicit bounded word time expansion required')
        self.expansion = expansion
        for name in ('norm', 'local', 'postnorm', 'height', 'temporal', 'classifier'):
            setattr(self, name, getattr(previous, name))

    def forward(self, patches):
        if patches.ndim != 4 or patches.shape[-1] != self.norm.weight.numel():
            raise ValueError('Whole word patch grid must be [batch,height,width,visual]')
        x = self.norm(patches).permute(0, 3, 1, 2)
        x = self.postnorm(self.local(x).permute(0, 2, 3, 1))
        sequence = (x * self.height(x).softmax(1)).sum(1)
        b, width, channels = sequence.shape
        sequence = sequence[:, :, None, :].expand(b, width, self.expansion, channels).reshape(b, width*self.expansion, channels)
        # Keep the own small recurrent readout and classifier in FP32 under
        # convolution/vision AMP; no external OCR weights or encoder is used.
        with torch.autocast(device_type=patches.device.type, enabled=False):
            features, _ = self.temporal(sequence.float())
            return self.classifier(features)


class OwnAlignedAssistantV2(OwnAlignedAssistant):
    def __init__(self, original, *, word_frame_expansion=5, **kwargs):
        if type(word_frame_expansion) is not int or not 1 <= word_frame_expansion <= 8:
            raise ValueError('Explicit bounded word time expansion required')
        super().__init__(original, **kwargs)
        # Re-register precisely the same new head parameters, replacing the
        # module rather than retaining an additional uncounted word network.
        self.parts['word'] = ExpandedByteWordHead(self.parts['word'], word_frame_expansion)
        self.settings = {**self.settings, 'word_frame_expansion': word_frame_expansion}
        self.assert_word_contract()

    def assert_word_contract(self):
        self.assert_shared()
        if (type(self.parts['word']) is not ExpandedByteWordHead
                or self.parts['word'].expansion != self.settings['word_frame_expansion']):
            raise ValueError('Exact source-bound expanded word computation required')

    def word_logits(self, pixel_values):
        self.assert_word_contract()
        return super().word_logits(pixel_values)

    def export(self, sources):
        self.assert_word_contract()
        return {**super().export(sources), 'format': FORMAT}

    def validate_export(self, payload, *, expected_sources):
        self.assert_word_contract()
        if payload.get('format') != FORMAT:
            raise ValueError('Exact expanded-word v2 computation format required')
        super().validate_export({**payload, 'format': PRIOR_FORMAT}, expected_sources=expected_sources)
