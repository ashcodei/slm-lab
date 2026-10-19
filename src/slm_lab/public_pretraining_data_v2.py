"""Add independently admitted whole-function CE to full raw/replay pretraining."""
import hashlib
import json

from .codenet_token_pack_v1 import CompleteCodePack
from .corpus import file_hash
from .focused_code_continuation_data_v1 import collate_complete
from .public_pretraining_data_v1 import PublicPretrainingData as PreviousData,TokenBudgetSchedule,RAW

FUNCTIONS='new:verified_original_Git_function_contracts'


def attach_function_metadata(data,pack):
    """Index only the train role; admission is checked by the public constructor."""
    if pack.role!='train' or FUNCTIONS in data.lengths:
        raise ValueError('A single sealed source-train function family is required')
    data.functions=pack
    entries=pack.index[pack.positions]
    data.lengths[FUNCTIONS]=[int(n) for n in entries['length']]
    data.targets[FUNCTIONS]=[int(a)-int(b) for a,b in zip(entries['length'],entries['target_start'])]
    data.kinds[FUNCTIONS]='text';data.populations[FUNCTIONS]=len(pack)
    if not len(pack) or max(data.lengths[FUNCTIONS])>4096 or min(data.targets[FUNCTIONS])<2:
        raise ValueError('Every source-train whole function and original EOS must remain present')
    data.retention_entries.append((FUNCTIONS,list(range(len(pack)))))
    data.retention_identity_sha256=hashlib.sha256(json.dumps(dict(previous=data.retention_identity_sha256,
        function_manifest_sha256=file_hash(pack.folder/'manifest.json'),original_train_positions=pack.positions.tolist()),sort_keys=True).encode()).hexdigest()


def function_batch(data,positions,device='cpu'):
    if any(type(p) is not int or not 0<=p<data.populations[FUNCTIONS] for p in positions):
        raise ValueError('Exact admitted source-train function positions required')
    if not positions:return None
    result=collate_complete([data.functions.sequence(p) for p in positions],device)
    if (int(result['attention_mask'].sum())!=sum(data.lengths[FUNCTIONS][p] for p in positions)
            or int((result['labels'][:,1:]!=-100).sum())!=sum(data.targets[FUNCTIONS][p] for p in positions)):
        raise ValueError('Every original function context, response mask and EOS must reach training')
    return result


class PublicPretrainingData(PreviousData):
    def __init__(self,root,config):
        from pathlib import Path
        root=Path(root).resolve();folder=root/'data/direct-algorithm-function-contracts-admitted-v1'
        receipt_path=root/'reports/direct-algorithm-function-contracts-v1-readback-admission/complete.json'
        receipt=json.loads(receipt_path.read_text());admission=json.loads((folder/'admission.json').read_text())
        if (receipt.get('complete') is not True or receipt.get('dataset_admitted_to_function_CE_training') is not True
                or receipt['actual_verified_roles']['train_functions']!=826 or receipt['actual_verified_roles']['valid_functions']!=15
                or receipt['source_files_sha256'].get(str(folder/'admission.json'))!=file_hash(folder/'admission.json')
                or admission.get('execution_reward_training_admitted') is not False
                or admission.get('human_preference_training_admitted') is not False):
            raise ValueError('Complete original-Git/native/fresh-public-example function CE admission required')
        functions=CompleteCodePack(folder,'train','training')
        super().__init__(root,config)
        if file_hash(folder/'tokenizer.json')!=self.code.provenance['tokenizer_sha256']:
            raise ValueError('The same own BPE is required for functions, raw text and every retained source')
        attach_function_metadata(self,functions)
        for path in [receipt_path,folder/'manifest.json',folder/'admission.json']:
            self.source_files_sha256[str(path)]=file_hash(path)
        self.source_files_sha256.update({str(folder/name):sha for name,sha in functions.manifest['files_sha256'].items()})
        self.source_files_sha256.update(receipt['source_files_sha256'])

    def batch(self,name,positions,device='cpu'):
        return function_batch(self,positions,device) if name==FUNCTIONS else super().batch(name,positions,device)
