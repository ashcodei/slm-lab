"""Observe exact live neural/source owners over45seconds, without signals."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/full-shard-original-jobs-observation-CPU-v1.json'
OWNERS=[dict(pid=1241,ticks='49351203'),dict(pid=18545,ticks='50128787'),
    dict(pid=25240,ticks='51635952'),dict(pid=34432,ticks='52786853')]


def snapshot():
    processes=[]
    for owner in OWNERS:
        v=(Path('/proc')/str(owner['pid'])/'stat').read_text().split(') ',1)[1].split()
        if v[19]!=owner['ticks'] or v[0] in {'Z','X','T','t'}:raise ValueError('Every exact original neural/source owner must remain live and unstopped')
        processes.append(dict(owner=owner,state=v[0],CPU_seconds=(int(v[11])+int(v[12]))/os.sysconf('SC_CLK_TCK'),nice=int(v[16])))
    path=ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl';raw=path.read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
    rows=[json.loads(x) for x in prefix.split(b'\n') if x]
    if [r['step'] for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('Contiguous full main update prefix required')
    progress=json.loads((ROOT/'reports/full-public-code-shard-qualification-CPU-v1/progress.json').read_text())
    if (ROOT/'reports/full-public-code-shard-qualification-CPU-v1/failure.json').exists():raise ValueError('The whole-source producer has failed')
    return dict(at=datetime.now(timezone.utc).isoformat(),monotonic=time.monotonic(),processes=processes,
        actual_main_updates=len(rows),actual_main_targets=sum(int(r['response_EOS_targets']) for r in rows),source_progress=progress)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists():raise ValueError('Fresh low-priority CPU observation required')
    first=snapshot();time.sleep(45);second=snapshot();wall=second['monotonic']-first['monotonic']
    cores={str(a['owner']['pid']):(b['CPU_seconds']-a['CPU_seconds'])/wall for a,b in zip(first['processes'],second['processes'])}
    if second['actual_main_updates']<=first['actual_main_updates'] or cores['34432']<=0:raise ValueError('Both actual main progress and actual source CPU work must be observed')
    value=dict(complete=True,first=first,second=second,wall_seconds=wall,actual_average_CPU_cores=cores,
        actual_main_update_delta=second['actual_main_updates']-first['actual_main_updates'],
        actual_main_target_delta=second['actual_main_targets']-first['actual_main_targets'],
        every_exact_original_owner_live_and_unmodified=True,new_neural_training_jobs=0,new_production_optimizer_updates_from_this_work=0,
        data_qualification_is_not_model_quality=True,goal_achieved=False)
    with OUT.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(complete=True,wall_seconds=wall,actual_main_update_delta=value['actual_main_update_delta'],source_worker_CPU_cores=cores['34432'])))


if __name__=='__main__':main()
