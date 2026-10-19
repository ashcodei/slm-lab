"""Compose existing packed corpora without pretending repeated tokens are unique."""
import argparse
import json
from pathlib import Path
import numpy as np
from .corpus import file_hash


def mix(components,output):
    root=Path(output).resolve()
    if root.exists() and any(root.iterdir()):raise ValueError('Use a new mixture directory')
    root.mkdir(parents=True,exist_ok=True)
    result={'components':[],'splits':{s:{'tokens':0,'shards':[]} for s in ['train','valid']},
            'mixture_rule':'Concatenate unique component tokens; training permutes packed windows. No oversampling.',
            'limitations':['Deduplication is within each component; cross-source near-duplicate removal remains pending.']}
    for name,path,train_limit in components:
        component=Path(path).resolve();manifest=json.loads((component/'corpus.json').read_text())
        if 'tokenizer_sha256' in result:
            if manifest['tokenizer_sha256']!=result['tokenizer_sha256'] or manifest['dtype']!=result['dtype']:
                raise ValueError('Components need the same tokenizer and token storage dtype')
        else:result.update({k:manifest[k] for k in ['tokenizer_sha256','dtype']})
        record={'name':name,'path':str(component),'manifest_sha256':file_hash(component/'corpus.json'),
                'source':manifest['source'],'domain_tokens':manifest.get('domain_tokens',{}),'included_tokens':{}}
        fraction=min(1,train_limit/manifest['splits']['train']['tokens']) if train_limit is not None else 1
        for split in ['train','valid']:
            budget=int(manifest['splits'][split]['tokens']*fraction)
            total=0
            for i,shard in enumerate(manifest['splits'][split]['shards']):
                count=min(shard['tokens'],budget-total)
                if count<=0:break
                origin=component/shard['path'];target=root/f'{name}-{split}-{i:05d}.bin'
                if count==shard['tokens']:target.symlink_to(origin)
                else:
                    values=np.memmap(origin,dtype=manifest['dtype'],mode='r')
                    values[:count].tofile(target)
                item={'path':target.name,'tokens':count,'sha256':file_hash(target),'component':name}
                result['splits'][split]['shards'].append(item);total+=count
            result['splits'][split]['tokens']+=total;record['included_tokens'][split]=total
        result['components'].append(record)
    result['source']={'mixture':[c['name'] for c in result['components']]}
    result['total_tokens']=sum(s['tokens'] for s in result['splits'].values())
    train=result['splits']['train']['tokens']
    result['train_token_shares']={c['name']:c['included_tokens']['train']/train for c in result['components']}
    (root/'corpus.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--general',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    code=json.loads((Path(args.code)/'corpus.json').read_text())['splits']['train']['tokens']
    reference=json.loads((Path(args.reference)/'corpus.json').read_text())['splits']['train']['tokens']
    # All available code and all domain references. General data fills to ~50% code,
    # without repeating scarce code merely to manufacture a larger corpus.
    general_limit=max(4096,code-reference)
    result=mix([('code',args.code,None),('reference',args.reference,None),
                ('general',args.general,general_limit)],args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='splits'},indent=2))

if __name__=='__main__':main()
