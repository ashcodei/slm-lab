"""One call-local visual prefix for native caption CE and paired matching.

The established two-forward objective remains untouched. This experimental
view reuses the same differentiable image features in its two loss branches.
It adds no parameters, persistent caches, model monkeypatches or source edits.
CPU parity tests are not full-size/GPU admission or evidence of better vision.
"""
import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .own_actor_vision_alignment_v1 import OwnActorVisionAlignment, symmetric_pair_loss
from .own_dense_actor_v1 import DensePolicyView


def native_from_visual(model, input_ids, visual, *, attention_mask=None, labels=None,
                       return_hidden=False, return_logits=True):
    """Run the own native decoder on an explicit, attached call-local prefix.

    This follows the existing native attention/position/loss semantics. It is
    deliberately a separate function: no frozen or live model file is edited,
    and no module or forward hook is temporarily replaced on the source model.
    """
    if (type(model) is not DensePolicyView or input_ids.ndim != 2
            or not 1 <= input_ids.shape[1] <= model.config.max_text_length
            or visual.ndim != 3 or visual.shape[0] != input_ids.shape[0]
            or visual.shape[1] < 1 or visual.shape[2] != model.config.hidden_size
            or visual.device != input_ids.device or not visual.is_floating_point()
            or not bool(torch.isfinite(visual).all())):
        raise ValueError('Exact own native policy and corresponding finite full visual prefix required')
    if torch.is_grad_enabled() and not visual.requires_grad:
        raise ValueError('Training must retain the actual visual prefix autograd connection')
    if labels is not None and not return_logits:
        raise ValueError('Token supervision requires logits')
    b,t = input_ids.shape; prefix = visual.shape[1]
    x = torch.cat((visual,model.embedding(input_ids)),dim=1)
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool:
            raise ValueError('Exact boolean text attention mask required')
        if bool(attention_mask.all()): attention_mask = None
    size = t+prefix
    mask = torch.ones(size,size,device=x.device,dtype=torch.bool).tril()
    mask[:prefix,:prefix] = True
    if attention_mask is not None:
        valid = torch.cat((torch.ones(b,prefix,device=x.device,dtype=torch.bool),attention_mask),dim=1)
        mask = mask[None,None,:,:] & valid[:,None,None,:]
    for block in model.blocks:
        if model.gradient_checkpointing and model.training:
            x = checkpoint(block,x,mask,False,use_reentrant=False)
        else: x = block(x,mask,False)
    hidden = model.norm(x[:,prefix:]); out = dict(values=model.critic(hidden))
    if return_logits: out['logits'] = model.lm_head(hidden)
    if return_hidden: out['hidden'] = hidden
    if labels is not None:
        if labels.shape != input_ids.shape:
            raise ValueError('Every complete original native target and EOS must retain the text shape')
        target = labels[:,1:].contiguous()
        if t < 2 or not bool((target != -100).any()):
            raise ValueError('At least one original next-token target required')
        out['loss'] = F.cross_entropy(out['logits'][:,:-1].float().reshape(-1,model.config.vocab_size),
                                      target.reshape(-1),ignore_index=-100)
    return out


class OwnActorVisionPrefixReuse(OwnActorVisionAlignment):
    def objective(self, caption_batch, text_ids, text_mask, positive, *, matching_weight=.1, temperature=.07):
        if (type(matching_weight) not in (int,float) or not math.isfinite(matching_weight) or matching_weight <= 0
                or caption_batch.get('pixel_values') is None or caption_batch.get('labels') is None
                or set(caption_batch) - {'input_ids','pixel_values','attention_mask','labels'}):
            raise ValueError('Complete native supervision and explicit positive matching weight required')
        pixels = caption_batch['pixel_values']; labels = caption_batch['labels']
        if (text_ids.ndim != 2 or text_mask.shape != text_ids.shape or text_mask.dtype != torch.bool
                or len(text_ids) != len(pixels) or len(labels) != len(pixels)
                or not bool(text_mask.any(1).all()) or bool((text_mask[:,1:] & ~text_mask[:,:-1]).any())):
            raise ValueError('Complete corresponding nonempty right-padded original captions required')
        targets = labels[:,1:]
        if (not bool((targets != -100).any(1).all())
                or any(int(row[row != -100][-1]) != 2 for row in targets)):
            raise ValueError('Every complete caption needs its genuine final EOS target')
        content = text_mask & (text_ids != 0) & (text_ids != 1) & (text_ids != 2)
        if not bool(content.any(1).all()):
            raise ValueError('Every matching caption must contain actual source content')
        with torch.no_grad():
            hidden = self.actor.policy(text_ids,attention_mask=text_mask,return_hidden=True,return_logits=False)['hidden']
            text = (hidden.float()*content[:,:,None]).sum(1)/content.sum(1,keepdim=True)
        # The exact same graph feeds caption CE and image/text matching. No
        # features are detached, recomputed, retained across updates or stored.
        visual = self.native.vision(pixels,self.native.gradient_checkpointing)
        if visual.shape != (len(pixels),self.actor.config.image_token_count(pixels.shape[-1]),self.actor.config.hidden_size):
            raise ValueError('Complete original image geometry and the actual native prefix must agree')
        heads = self.actor.base.heads['pointing']
        images = heads['patch'](visual.mean(1)); texts = heads['query'](text.to(visual.dtype))
        matching = symmetric_pair_loss(images,texts,positive,temperature=temperature)
        batch = {k:v for k,v in caption_batch.items() if k != 'pixel_values'}
        caption = native_from_visual(self.native,visual=visual,**batch)['loss']
        return dict(loss=caption+matching_weight*matching,caption_CE=caption,matching_loss=matching,
                    response_EOS_targets=int((targets != -100).sum()))
