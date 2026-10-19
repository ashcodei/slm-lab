"""Exact union of independently admitted Java/Python publisher execution views."""
import json
from pathlib import Path

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .execution_feedback_reader_v1 import ExecutionFeedback
from .python_feedback_reader_v2 import PythonExecutionFeedbackV2

FORMAT='own-published-code-execution-feedback-union-v1'
ADMISSION_FORMAT='own-finite-published-code-feedback-union-admission-v1'


def merge_files(files,added):
    for name,digest in added.items():
        name=str(name)
        if name in files and files[name]!=digest:raise ValueError('Conflicting original parent evidence binding')
        files[name]=digest


def parents(manifest,tokenizer,tokenizer_path):
    if set(manifest.get('parents',{}))!={'Java','Python'}:raise ValueError('Both original Java and Python feedback parents required')
    result=[]
    for name,reader in [('Java',ExecutionFeedback),('Python',PythonExecutionFeedbackV2)]:
        item=manifest['parents'][name];folder=Path(item['folder'])
        if (not folder.is_absolute() or file_hash(folder/'manifest.json')!=item['manifest_sha256']
                or file_hash(folder/'admission.json')!=item['admission_sha256']):raise ValueError('Exact original independently admitted parent bytes required')
        result.append(reader(folder,tokenizer,tokenizer_path))
    return result


class ExecutionFeedbackUnion(ExecutionFeedback):
    """Full source interface consumed by the unchanged complete execution kernel."""
    def __init__(self,folder,tokenizer,tokenizer_path):
        self.folder=Path(folder).resolve();self.tokenizer=tokenizer
        self.manifest=json.loads((self.folder/'manifest.json').read_text())
        self.admission=json.loads((self.folder/'admission.json').read_text())
        if (self.manifest.get('format')!=FORMAT or self.admission.get('format')!=ADMISSION_FORMAT
                or self.admission.get('passed') is not True or self.admission.get('source_roles_only_train') is not True
                or self.admission.get('machine_feedback_not_human_preferences') is not True
                or self.admission.get('candidate_manifest_sha256')!=file_hash(self.folder/'manifest.json')
                or self.admission.get('all_parent_views_retained_exactly') is not True
                or self.admission.get('held_group_overlap')!=0):raise ValueError('Complete independently admitted original Java/Python feedback union required')
        token_path=Path(tokenizer_path).resolve()
        if file_hash(token_path)!=self.manifest['tokenizer_sha256']:raise ValueError('Exact original own union tokenizer required')
        self.owners=parents(self.manifest,tokenizer,token_path)
        self.pairs=list(map(json.loads,(self.folder/'pairs.jsonl').read_text().splitlines()))
        self.positives=list(map(json.loads,(self.folder/'passing-originals.jsonl').read_text().splitlines()))
        for key in ['pairs','positives']:
            expected=[r for owner in self.owners for r in getattr(owner,key)];actual=getattr(self,key)
            if (actual!=expected or len(actual)!=len({r['id'] for r in actual})
                    or any(r['split']!='train' or r['human_preferences'] is not False for r in actual)):
                raise ValueError('Every complete parent view must remain byte-equivalent and in its original role')
        if (len(self.pairs)!=self.admission['finite_assertion_pairs']
                or len(self.positives)!=self.admission['complete_passing_originals']
                or {r['language'] for r in self.pairs}!={'Java','Python'}):raise ValueError('Actual complete two-language populations required')
        files=dict(self.admission['source_files_sha256'])
        for owner in self.owners:merge_files(files,{str(p):h for p,h in owner.guard.files.items()})
        merge_files(files,{str(self.folder/'admission.json'):file_hash(self.folder/'admission.json'),
            str(Path(__file__).resolve()):file_hash(__file__),str(token_path):file_hash(token_path)})
        self.guard=FileBindings(files)
