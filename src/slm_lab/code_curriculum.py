"""Derive bounded function-completion/SFT pairs from licensed training-only Python.
Source modules are parsed with ast; downloaded code is never executed.
"""
import argparse
import ast
import builtins
import hashlib
import json
from pathlib import Path
import re
from .data import stable_id
from .corpus import file_hash


def extract(row):
    if row.get('language_extension')!='.py' or row.get('chunk',0)!=0:return []
    # File/language headers were added by our collector, not part of Python.
    text=re.sub(r'^File: .*?\nLanguage: .*?\n\n','',row['text'],count=1)
    try:tree=ast.parse(text)
    except SyntaxError:return []
    functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    imports=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
    constants={}
    for statement in tree.body:
        if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):
            try:ast.literal_eval(statement.value)
            except (ValueError,TypeError,RecursionError):continue
            segment=ast.get_source_segment(text,statement)
            if segment and len(segment)<1000:constants[statement.targets[0].id]=segment
    pairs=[]
    for name,node in functions.items():
        description=ast.get_docstring(node)
        if not description or len(description)<30 or len(description)>1800 or node.decorator_list:continue
        if node.end_lineno-node.lineno>100:continue
        if any(isinstance(child,(ast.Global,ast.Nonlocal)) for child in ast.walk(node)):continue
        loaded={n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
        # Helpers/complex module state are outside this initial function curriculum.
        if (loaded&set(functions))-{name}:continue
        relevant=[];imported=set()
        for imp in imports:
            defined={(n.asname or n.name.split('.')[0]) for n in imp.names}
            if loaded&defined:
                relevant.append(ast.get_source_segment(text,imp));imported.update(defined)
        locals_={n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        locals_.update(n.arg for n in ast.walk(node) if isinstance(n,ast.arg))
        locals_.update(n.name for n in ast.walk(node) if isinstance(n,(ast.FunctionDef,ast.ClassDef)))
        for variable in loaded&set(constants):relevant.append(constants[variable])
        unresolved=loaded-locals_-imported-set(constants)-set(dir(builtins))-{name}
        if unresolved:continue
        code=ast.get_source_segment(text,node)
        if not code or len(code)>12000:continue
        signature=ast.unparse(node.args)
        return_type=' -> '+ast.unparse(node.returns) if node.returns else ''
        specification=f'Implement this Python function:\n\ndef {name}({signature}){return_type}:\n\n{description}'
        response='```python\n'+'\n'.join(relevant)+('\n\n' if relevant else '')+code+'\n```'
        key=stable_id(str(row['group_id'])+name)
        pairs.append({'id':key,'group_id':row['group_id'],'prompt':specification,'response':response,
            'language':'Python','source':row['source'],'source_url':row['source_url'],
            'revision':row['revision'],'license':row['license'],'license_url':row.get('license_url'),
            'annotation_origin':'Human-authored source docstring and implementation; statically transformed, not preference feedback.',
            'function':name,'code_sha256':hashlib.sha256(code.encode()).hexdigest()})
    return pairs


def build(paths,output,max_pairs=12000):
    root=Path(output)
    if root.exists() and any(root.iterdir()):raise ValueError('Use a new output directory.')
    root.mkdir(parents=True,exist_ok=True);train=[];valid=[];seen=set();origins={}
    for path in paths:
        with open(path) as source:
            for line in source:
                row=json.loads(line)
                for pair in extract(row):
                    if pair['code_sha256'] in seen:continue
                    seen.add(pair['code_sha256'])
                    split='valid' if int(stable_id(str(pair['group_id']))[:8],16)%10==0 else 'train'
                    target=valid if split=='valid' else train
                    if len(target)>=(max_pairs//9 if split=='valid' else max_pairs):continue
                    target.append(pair);origins[pair['source']]=origins.get(pair['source'],0)+1
    if not train or not valid:raise ValueError('Need source-group separated training and validation functions.')
    for split,rows in [('train',train),('valid',valid)]:
        (root/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest={'train_pairs':len(train),'valid_pairs':len(valid),'source_pairs':origins,
        'inputs':[{'path':str(Path(p).resolve()),'sha256':file_hash(p)} for p in paths],
        'files_sha256':{s+'.jsonl':file_hash(root/(s+'.jsonl')) for s in ['train','valid']},
        'split_rule':'Same source-file SHA256 modulo 10 as pretraining; validation files stay out of SFT.',
        'limitations':['Only selected top-level Python functions with docstrings.',
            'Implementations have not been executed or verified; docstrings can be incomplete.',
            'Other languages, repository-level tasks, tests and dependency handling need richer curricula.']}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',action='append',required=True)
    p.add_argument('--output',required=True);p.add_argument('--max-pairs',type=int,default=12000)
    a=p.parse_args();build(a.input,a.output,a.max_pairs)

if __name__=='__main__':main()
