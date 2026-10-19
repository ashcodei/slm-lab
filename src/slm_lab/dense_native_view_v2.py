"""Expose the actual owned vision module to native retrieval evaluators."""
from .dense_foundation_inference_v1 import DenseNativeView


class CompleteDenseNativeView(DenseNativeView):
    @property
    def vision(self):return self.assistant.network.vision
