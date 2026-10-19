"""Require held-out human-ranking evidence before a bounded PPO-RLHF experiment."""
import math


def wilson(correct,count,z=1.959963984540054):
    if count<1 or not 0<=correct<=count:raise ValueError('Invalid ranking counts')
    p=correct/count;denominator=1+z*z/count
    center=(p+z*z/(2*count))/denominator
    radius=z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [center-radius,center+radius]


def gate(report,minimum_examples=500):
    if report.get('task')!='preference':raise ValueError('Gate requires reward-model ranking evaluation')
    count=report['examples'];correct=report['correct'];interval=wilson(correct,count)
    failures=[]
    if count<minimum_examples:failures.append('Insufficient held-out human preference pairs')
    if interval[0]<=.5:failures.append('95% ranking accuracy interval does not exceed chance')
    if not report.get('checkpoint_sha256') or not report.get('data_sha256'):failures.append('Checkpoint/data provenance missing')
    if set(report.get('feedback_sources',[]))!={'human'}:failures.append('Training feedback is not exclusively identified as human')
    if report.get('truncated_sequences',0):failures.append('Some validation prompt/completion text was truncated')
    return {'passed':not failures,'examples':count,'correct':correct,'accuracy':correct/count,
            'accuracy_95ci':interval,'ties':report.get('ties'),'failures':failures,
            'scope':'Offline human preference ranking gate for a PPO pilot; not proof of on-policy reward fidelity.',
            'pending':['Fresh human evaluation of this model\u2019s sampled responses','Checks for reward exploitation and domain transfer']}
