"""Read independently admitted full-program Python machine-execution feedback."""
import json
from pathlib import Path

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .execution_feedback_reader_v1 import ExecutionFeedback


class PythonExecutionFeedbackV2(ExecutionFeedback):
    """The same complete-sequence interface accepted by ExecutionResidualKernel."""
    def __init__(self,folder,tokenizer,tokenizer_path):
        self.folder=Path(folder).resolve();self.tokenizer=tokenizer
        self.manifest=json.loads((self.folder/'manifest.json').read_text())
        self.admission=json.loads((self.folder/'admission.json').read_text())
        if (self.manifest.get('format')!='own-published-Python-execution-feedback-candidate-v2'
                or self.admission.get('format')!='own-finite-published-Python-feedback-admission-v2'
                or self.admission.get('passed') is not True or self.admission.get('source_roles_only_train') is not True
                or self.admission.get('candidate_manifest_sha256')!=file_hash(self.folder/'manifest.json')
                or self.admission.get('machine_feedback_not_human_preferences') is not True
                or self.admission.get('original_sources_accounted')!=86
                or self.admission.get('held_group_overlap')!=0):
            raise ValueError('Full original train-only Python publisher feedback admission required')
        token_path=Path(tokenizer_path).resolve()
        if (file_hash(token_path)!=self.manifest['tokenizer_sha256']
                or json.loads(tokenizer.to_str())!=json.loads(token_path.read_text())
                or tokenizer.token_to_id('<bos>')!=1 or tokenizer.token_to_id('<eos>')!=2
                or tokenizer.token_to_id('<pad>')!=0 or tokenizer.get_vocab_size()!=32768):
            raise ValueError('Exact original own BPE bytes, semantics and reserved token IDs required')
        bindings={**self.admission['source_files_sha256'],str(self.folder/'admission.json'):file_hash(self.folder/'admission.json'),
            str(token_path):self.manifest['tokenizer_sha256'],str(Path(__file__).resolve()):file_hash(__file__),
            str(Path(__file__).with_name('execution_feedback_reader_v1.py')):file_hash(Path(__file__).with_name('execution_feedback_reader_v1.py'))}
        self.guard=FileBindings(bindings)
        self.pairs=list(map(json.loads,(self.folder/'pairs.jsonl').read_text().splitlines()))
        self.positives=list(map(json.loads,(self.folder/'passing-originals.jsonl').read_text().splitlines()))
        if (not self.pairs or not self.positives or len(self.pairs)!=self.admission['finite_assertion_pairs']
                or len(self.positives)!=self.admission['complete_passing_originals']):
            raise ValueError('Every admitted complete Python source view required')
        for rows in [self.pairs,self.positives]:
            if len({r['id'] for r in rows})!=len(rows) or any(r['split']!='train' or r['language']!='Python'
                    or r['human_preferences'] is not False for r in rows):
                raise ValueError('Distinct original train-only Python machine feedback required')
