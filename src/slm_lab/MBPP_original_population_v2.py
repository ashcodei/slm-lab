"""All reserved original publisher MBPP cases and a fixed native prompt.

Public function/class signatures come from the original interface. No source
body, original test inputs, expected values or reference executions are placed
in the model prompt. All600 problems remain; publisher roles stay separate.
"""
import ast
import hashlib
import json
from pathlib import Path

from .corpus import file_hash
from .mbpp_original_function_cases_v2 import cases
from .source_jsonl_records_v1 import records

BUDGET=512
FORMAT='own-all600-original-MBPP-reserved-evaluation-v2'


def role(task_id):
    if type(task_id) is not int or not 1<=task_id<=600:raise ValueError('An original reserved task1through600required')
    return 'publisher_prompting' if task_id<=10 else ('publisher_test' if task_id<=510 else 'publisher_validation')


def prompt(row):
    original=cases(row);tree=ast.parse(row['code']);used=set()
    for source in [row.get('test_setup_code',''),*[c['left_expression'] for c in original]]:
        used.update(n.id for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load))
    signatures=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in used:
            signatures.append('def '+node.name+'('+ast.unparse(node.args)+'):')
        elif isinstance(node,ast.ClassDef) and node.name in used:
            signatures.append('class '+node.name+':')
            for method in node.body:
                if isinstance(method,ast.FunctionDef) and method.name=='__init__':
                    signatures.append('    def __init__('+ast.unparse(method.args)+'):')
    if not signatures:raise ValueError('Every original tested public function or constructor must be represented without source body')
    attrs=sorted({n.attr for n in ast.walk(ast.parse(row.get('test_setup_code',''))) if isinstance(n,ast.Attribute)})
    hint='\nInput objects use these attributes: '+', '.join(attrs)+'.' if attrs else ''
    return ('User: Write complete Python 3 code for this task.\n'+row['text']+'\nRequired public interface:\n'
        +'\n'.join(signatures)+hint+'\nUse the Python standard library. Return only complete code.\nAssistant:')


class FullOriginalMBPP:
    purpose='evaluation_only'
    optimization_forbidden=True

    def __init__(self,root):
        from tokenizers import Tokenizer
        root=Path(root).resolve();self.root=root;self.folder=root/'data/MBPP-all-reserved-original-evaluation-v2'
        manifest_path=self.folder/'manifest.json';self.manifest=json.loads(manifest_path.read_text())
        if self.manifest.get('format')!=FORMAT or self.manifest.get('original_problems')!=600 or self.manifest.get('optimization_forbidden') is not True:
            raise ValueError('Full original600problem evaluation-only population required')
        for name,digest in self.manifest['files_sha256'].items():
            if file_hash(self.folder/name)!=digest:raise ValueError('Every original MBPP source/input/control artifact must remain exact')
        self.rows=list(records(self.folder/'original-reserved-rows.jsonl'))
        self.inputs=list(records(self.folder/'original-native-inputs.jsonl'))
        self.controls=json.loads((self.folder/'all-original-reference-controls.json').read_text())
        if ([r['task_id'] for r in self.rows]!=list(range(1,601))
                or [r['original_task_id'] for r in self.inputs]!=list(range(1,601))
                or len(self.controls)!=600 or [r['original_task_id'] for r in self.controls]!=list(range(1,601))):
            raise ValueError('No original reserved problem, control or input may be dropped or reordered')
        self.tokenizer_path=root/'data/tokenizer-corpus/tokenizer.json';self.tokenizer_sha256=file_hash(self.tokenizer_path)
        if self.tokenizer_sha256!=self.manifest['tokenizer_sha256']:raise ValueError('Exact own native BPE required')
        self.tokenizer=Tokenizer.from_file(str(self.tokenizer_path))
        self.identity_sha256=self.manifest['original_population_sha256']
        expected=hashlib.sha256(json.dumps(self.rows,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        if expected!=self.identity_sha256:raise ValueError('Every original publisher field must remain unchanged')
        self.bindings={str(manifest_path):file_hash(manifest_path),str(self.tokenizer_path):self.tokenizer_sha256,
            **{str(self.folder/name):digest for name,digest in self.manifest['files_sha256'].items()},
            str(root/'reports/MBPP-all-reserved-original-v2-preparation/complete.json'):file_hash(root/'reports/MBPP-all-reserved-original-v2-preparation/complete.json')}
