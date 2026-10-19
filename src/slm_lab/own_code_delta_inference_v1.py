"""Load one own general backbone and bounded own code components, without a teacher."""
from dataclasses import asdict
from pathlib import Path
import re

import torch
from tokenizers import Tokenizer

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .own_code_delta_v1 import OwnCodeDelta
from .training import load_checkpoint


def load(checkpoint,base_checkpoint,tokenizer,run_plan,*,expected_checkpoint_sha256,
         expected_base_sha256,expected_target_sha256,expected_plan_sha256,device='cpu'):
    if any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in [expected_checkpoint_sha256,
            expected_base_sha256,expected_target_sha256,expected_plan_sha256]):
        raise ValueError('Externally bound own component/base/target/provenance SHA256 required')
    checkpoint,base_checkpoint,tokenizer,run_plan=map(Path,[checkpoint,base_checkpoint,tokenizer,run_plan])
    guard=FileBindings({str(checkpoint):expected_checkpoint_sha256,str(base_checkpoint):expected_base_sha256,
        str(tokenizer):file_hash(tokenizer),str(run_plan):expected_plan_sha256})
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if (state['format']!='own-conditional-code-delta-v1' or state['base_checkpoint_sha256']!=expected_base_sha256
            or state['own_code_source_checkpoint_sha256']!=expected_target_sha256
            or state['source_plan_sha256']!=expected_plan_sha256 or state['tokenizer_sha256']!=file_hash(tokenizer)
            or state['candidate_promoted'] is not False or state['goal_achieved'] is not False):
        raise ValueError('Exact own complete code components and original provenance required')
    base,metadata=load_checkpoint(base_checkpoint,'cpu',training_state=False)
    if (asdict(base.config)!=state['config'] or metadata['tokenizer_sha256']!=file_hash(tokenizer)
            or Tokenizer.from_file(str(tokenizer)).get_vocab_size()!=base.config.vocab_size):
        raise ValueError('Exact original own model configuration and tokenizer required')
    model=OwnCodeDelta(base,state['components'],reserved_parameters=state['reserved_point_head_parameters'])
    counts=model.counts()
    if (counts['frozen_general']!=state['base_parameters'] or counts['trainable_code']!=state['additional_code_parameters']
            or counts['total_unique']!=state['total_unique_parameters']
            or counts['total_with_reserved_head']!=state['total_with_reserved_point_head']):
        raise ValueError('Actual full unique original/code/future-point parameter inventory differs')
    guard.check(full=True);model.eval().to(device);model.assert_partition()
    return model,dict(format=state['format'],component_checkpoint_sha256=expected_checkpoint_sha256,
        base_checkpoint_sha256=expected_base_sha256,own_code_source_checkpoint_sha256=expected_target_sha256,
        source_plan_sha256=expected_plan_sha256,tokenizer_sha256=state['tokenizer_sha256'],model_counts=counts,
        full_code_source_loaded_at_inference=False,external_model_loaded=False,
        explicit_general_and_code_routes=True,automatic_routing_unproven=True,candidate_promoted=False)
