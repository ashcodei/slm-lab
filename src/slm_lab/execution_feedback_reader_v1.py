"""Read independently admitted supplemental finite-execution training views."""
import hashlib
import json
from pathlib import Path

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .execution_code_feedback_v1 import sequence


class ExecutionFeedback:
    def __init__(self,folder,tokenizer,tokenizer_path):
        self.folder=Path(folder).resolve();self.tokenizer=tokenizer
        self.manifest=json.loads((self.folder/'manifest.json').read_text())
        self.admission=json.loads((self.folder/'admission.json').read_text())
        if (self.manifest['format']!='own-published-Java-execution-feedback-candidate-v1'
                or self.admission['format']!='own-finite-published-execution-feedback-admission-v1'
                or self.admission['passed'] is not True or self.admission['source_roles_only_train'] is not True
                or self.admission['candidate_manifest_sha256']!=file_hash(self.folder/'manifest.json')
                or self.admission['machine_feedback_not_human_preferences'] is not True
                or self.admission['original_sources_accounted']!=109):
            raise ValueError('Complete independent original-source/execution feedback admission required')
        token_path=Path(tokenizer_path).resolve()
        if (file_hash(token_path)!=self.manifest['tokenizer_sha256']
                or json.loads(tokenizer.to_str())!=json.loads(token_path.read_text())
                or tokenizer.token_to_id('<bos>')!=1 or tokenizer.token_to_id('<eos>')!=2
                or tokenizer.token_to_id('<pad>')!=0 or tokenizer.get_vocab_size()!=32768):
            raise ValueError('Exact admitted own BPE bytes, normalizer and reserved token IDs required')
        self.guard=FileBindings({**self.admission['source_files_sha256'],
            str(self.folder/'admission.json'):file_hash(self.folder/'admission.json'),
            str(Path(__file__).resolve()):file_hash(__file__),str(token_path):self.manifest['tokenizer_sha256']})
        self.pairs=list(map(json.loads,(self.folder/'pairs.jsonl').open()))
        self.positives=list(map(json.loads,(self.folder/'passing-originals.jsonl').open()))
        if len(self.pairs)!=self.admission['finite_assertion_pairs'] or len(self.positives)!=self.admission['complete_passing_originals']:
            raise ValueError('Every admitted whole pair and original positive must remain available')
        for rows in [self.pairs,self.positives]:
            if any(r['split']!='train' or r['human_preferences'] is not False for r in rows):
                raise ValueError('Only explicit original-training machine feedback may be read')

    def example(self,position):
        self.guard.check()
        if type(position) is not int or not 0<=position<len(self.pairs):raise IndexError('Exact local admitted pair index required')
        row=self.pairs[position]
        for name in ['chosen','rejected']:
            if hashlib.sha256(row[name].encode()).hexdigest()!=row[name+'_sha256']:
                raise ValueError('Every admitted complete original/mutated source body must match')
        chosen=sequence(self.tokenizer,row['prompt'],row['chosen']);rejected=sequence(self.tokenizer,row['prompt'],row['rejected'])
        for name,value in [('chosen',chosen),('rejected',rejected)]:
            if len(value[0])!=row[name+'_context_tokens'] or sum(v!=-100 for v in value[1])!=row[name+'_target_tokens']:
                raise ValueError('Every complete original prompt/response/EOS/mask must match admission')
        return dict(id=row['id'],split='train',chosen=chosen,rejected=rejected,
            feedback_source=row['feedback_source'],human_preferences=False,finite_execution_verified=True)

    def positive(self,position):
        self.guard.check()
        if type(position) is not int or not 0<=position<len(self.positives):raise IndexError('Exact local admitted original index required')
        row=self.positives[position]
        if hashlib.sha256(row['response'].encode()).hexdigest()!=row['source_response_sha256']:
            raise ValueError('Original full admitted positive source changed')
        value=sequence(self.tokenizer,row['prompt'],row['response'])
        if len(value[0])!=row['context_tokens'] or sum(v!=-100 for v in value[1])!=row['target_tokens']:
            raise ValueError('Whole original positive tokens/mask must match')
        return value
