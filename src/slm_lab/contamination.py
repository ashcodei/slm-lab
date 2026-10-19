"""Audit benchmark question substrings against the actual packed training tokens."""
import argparse
import json
from pathlib import Path
import ahocorasick
import numpy as np
from tokenizers import Tokenizer
from .corpus import file_hash


def normalize(text):return ' '.join(text.casefold().split())


def audit(corpus,tokenizer,questions,output):
    root=Path(corpus);manifest=json.loads((root/'corpus.json').read_text())
    if manifest['tokenizer_sha256']!=file_hash(tokenizer):raise ValueError('Training tokenizer mismatch')
    tok=Tokenizer.from_file(str(tokenizer));rows=[json.loads(line) for line in Path(questions).read_text().splitlines()]
    patterns={}
    for row in rows:patterns.setdefault(normalize(row['question']),[]).append(row['id'])
    automaton=ahocorasick.Automaton()
    for text,identifiers in patterns.items():automaton.add_word(text,(text,identifiers))
    automaton.make_automaton();max_query=max(len(tok.encode(row['question']).ids) for row in rows)
    overlap=max_query+32;tail=[];matched={};tokens=0
    for shard in manifest['splits']['train']['shards']:
        path=root/shard['path']
        if file_hash(path)!=shard['sha256']:raise ValueError('Training shard checksum mismatch')
        values=np.memmap(path,dtype=manifest['dtype'],mode='r')
        for start in range(0,len(values),100000):
            ids=tail+values[start:start+100000].astype(np.int64).tolist()
            # Keep document boundaries as whitespace rather than fusing words.
            text=normalize(tok.decode(ids,skip_special_tokens=False).replace('<eos>',' ').replace('<bos>',' '))
            for _,(_,identifiers) in automaton.iter(text):
                for key in identifiers:matched.setdefault(key,shard['path'])
            tail=ids[-overlap:];tokens+=min(100000,len(values)-start)
        print('audited shard',shard['path'],'tokens',tokens,'matched questions',len(matched),flush=True)
    clean=[row for row in rows if row['id'] not in matched]
    Path(output).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in clean))
    report={'training_manifest_sha256':file_hash(root/'corpus.json'),'questions_sha256':file_hash(questions),
        'audited_training_tokens':tokens,'input_questions':len(rows),'retained_questions':len(clean),
        'excluded_exact_matches':matched,'clean_questions_sha256':file_hash(output),
        'scope':'Exact casefold/whitespace-normalized question substrings in actual training shards; near-duplicates and semantic overlap are not ruled out.'}
    Path(output).with_suffix('.audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus',required=True)
    p.add_argument('--tokenizer',required=True);p.add_argument('--questions',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.corpus,a.tokenizer,a.questions,a.output)

if __name__=='__main__':main()
