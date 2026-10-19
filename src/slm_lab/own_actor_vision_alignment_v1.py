"""Experimental native-caption and multi-positive image/text alignment.

Reuse one own actor, its image encoder and existing pointing projections.
No new neural values or external encoder/teacher are introduced. The language
computation stays live so caption gradients can reach the visual prefix;
only image and pointing parameters belong to the explicit optimizer set.
This is a training mechanism, not evidence of improved visual capability.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .own_dense_actor_v1 import DensePolicyView, OwnDenseActor


def pair_positives(pixel_keys, captions):
    """Use every observed pixel/caption association, without image transitivity.

    Each cell describes that image/caption pair. A caption is positive for all
    repeated rows of its exact image, including duplicates of that caption.
    Transitive grouping would incorrectly treat every caption associated with
    either image as an annotation of all images connected by a generic caption.
    Source text identity is conservative: no semantic-equivalence claim.
    """
    if (not isinstance(pixel_keys, (list, tuple)) or not isinstance(captions, (list, tuple))
            or len(pixel_keys) != len(captions) or len(pixel_keys) < 2
            or any(not isinstance(s, str) or not s.strip() for s in [*pixel_keys, *captions])):
        raise ValueError('At least two exact pixel keys and complete nonempty source captions required')
    observed = {}
    for pixels, text in zip(pixel_keys, captions):
        observed.setdefault(pixels, set()).add(text)
    return torch.tensor([[text in observed[pixels] for text in captions] for pixels in pixel_keys], dtype=torch.bool)


def symmetric_pair_loss(images, texts, positive, *, temperature=.07):
    if (images.ndim != 2 or images.shape != texts.shape or len(images) < 2
            or images.device != texts.device or not images.is_floating_point() or not texts.is_floating_point()
            or positive.dtype != torch.bool or positive.shape != (len(images), len(images))
            or type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0
            or not bool(torch.isfinite(images).all()) or not bool(torch.isfinite(texts).all())):
        raise ValueError('Finite paired features, exact boolean positives and positive finite temperature required')
    positive = positive.to(images.device)
    if (not bool(positive.diagonal().all()) or not bool(positive.any(0).all())
            or not bool(positive.any(1).all()) or bool(positive.all())):
        raise ValueError('Each direction needs observed positives and the batch needs some negatives')
    with torch.autocast(device_type=images.device.type, enabled=False):
        if bool((images.float().norm(dim=-1) == 0).any()) or bool((texts.float().norm(dim=-1) == 0).any()):
            raise ValueError('Nonzero features required for normalized matching')
        scores = F.normalize(images.float(), dim=-1) @ F.normalize(texts.float(), dim=-1).T / temperature
        row = torch.logsumexp(scores, -1) - torch.logsumexp(scores.masked_fill(~positive, -torch.inf), -1)
        col = torch.logsumexp(scores, 0) - torch.logsumexp(scores.masked_fill(~positive, -torch.inf), 0)
        return (row.mean() + col.mean()) / 2


class OwnActorVisionAlignment(nn.Module):
    def __init__(self, actor):
        super().__init__()
        if type(actor) is not OwnDenseActor:
            raise ValueError('Exactly one original own dense actor required')
        actor.assert_shared()
        self.actor = actor
        self.native = DensePolicyView(actor.base, actor.parts)
        # The ordinary actor intentionally detaches its vision branch. This
        # training view shares that exact encoder but enables its derivatives.
        self.native.vision = actor.base.network.vision
        self.assert_shared()

    def assert_shared(self):
        self.actor.assert_shared()
        if (self.native.vision is not self.actor.base.network.vision
                or self.native.parts is not self.actor.parts
                or {id(p) for p in self.parameters()} != {id(p) for p in self.actor.parameters()}
                or sum(p.numel() for p in self.parameters()) != self.actor.counts()['actual_unique']):
            raise ValueError('No additional neural values and exactly one shared own image encoder required')

    def named_selected(self):
        return [('vision.' + n, p) for n, p in self.actor.base.network.vision.named_parameters()] + [
            ('pointing.' + n, p) for n, p in self.actor.base.heads['pointing'].named_parameters()]

    def features(self, caption_ids, caption_mask, pixels):
        if (caption_ids.ndim != 2 or caption_mask.shape != caption_ids.shape
                or caption_mask.dtype != torch.bool or len(caption_ids) != len(pixels)
                or not bool(caption_mask.any(1).all())
                or bool((caption_mask[:, 1:] & ~caption_mask[:, :-1]).any())):
            raise ValueError('Complete nonempty right-padded original captions required')
        content = caption_mask & (caption_ids != 0) & (caption_ids != 1) & (caption_ids != 2)
        if not bool(content.any(1).all()):
            raise ValueError('Every caption must have actual source content')
        with torch.no_grad():
            hidden = self.actor.policy(caption_ids, attention_mask=caption_mask,
                                       return_hidden=True, return_logits=False)['hidden']
            text = (hidden.float() * content[:, :, None]).sum(1) / content.sum(1, keepdim=True)
        image = self.native.vision(pixels, self.native.gradient_checkpointing).mean(1)
        heads = self.actor.base.heads['pointing']
        return heads['patch'](image), heads['query'](text.to(image.dtype))

    def objective(self, caption_batch, text_ids, text_mask, positive, *, matching_weight=.1, temperature=.07):
        if (type(matching_weight) not in (int, float) or not math.isfinite(matching_weight)
                or matching_weight <= 0 or caption_batch.get('pixel_values') is None
                or caption_batch.get('labels') is None):
            raise ValueError('Native complete image-caption supervision and explicit positive matching weight required')
        targets = caption_batch['labels'][:, 1:]
        if (not bool((targets != -100).any(1).all())
                or any(int(row[row != -100][-1]) != 2 for row in targets)):
            raise ValueError('Every native caption needs its original response and genuine EOS target')
        images, texts = self.features(text_ids, text_mask, caption_batch['pixel_values'])
        matching = symmetric_pair_loss(images, texts, positive, temperature=temperature)
        caption = self.native(**caption_batch)['loss']
        return dict(loss=caption + matching_weight * matching, caption_CE=caption,
                    matching_loss=matching, response_EOS_targets=int((targets != -100).sum()))

    def backward(self, objective):
        # Explicit inputs prevent gradients/Adam clocks/decay on all language,
        # actor, critic, word, detector, reward and router parameters.
        parameters = [p for _, p in self.named_selected()]
        if any(p.grad is not None for p in self.parameters()):
            raise ValueError('Clear all gradients before each owned vision update')
        objective['loss'].backward(inputs=parameters)
        chosen = {id(p) for p in parameters}
        if (any(p.grad is not None for p in self.parameters() if id(p) not in chosen)
                or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters)):
            raise ValueError('Only the explicit image/pointing parameters may receive finite gradients')

