"""Bounded public-train input perturbations with reference-derived outcomes.

Original published examples stay unchanged. New probes are machine-generated,
not publisher examples or human labels. Agreement with a bounded original
implementation is not a formal input-domain or general-correctness certificate.
"""
import ast
import copy
import json
import math
from pathlib import Path
import tempfile

from .code_quality import instruction_code
from .documented_doctest_v1 import contract,compare,unresolved_globals,PRELUDE,HARNESS,sha
from .causal_code_reward_v2 import pass_ablation
from .standalone_program_v1 import python_policy,run_bounded

CONSTRUCTORS={'iter','range','list','tuple','set','dict','str','bytes'}
POOL=['0','1','-1','3','True','None','0.5',"''","'abc'","b'abc'",'[]','[0, 1]','()','{}','set()','iter([1])','range(3)',"'π'"]


def closed(node):
    if isinstance(node,ast.Constant):
        value=node.value
        if value is None or type(value) is bool:return True
        if type(value) in (int,float,complex):return abs(value)<=32 and (type(value) is int or math.isfinite(abs(value)))
        return type(value) in (str,bytes) and len(value)<=32
    if isinstance(node,(ast.List,ast.Tuple,ast.Set)):return len(node.elts)<=8 and all(closed(x) for x in node.elts)
    if isinstance(node,ast.Dict):return len(node.keys)<=8 and all(k is not None and closed(k) and closed(v) for k,v in zip(node.keys,node.values))
    if isinstance(node,ast.UnaryOp):return isinstance(node.op,(ast.UAdd,ast.USub)) and closed(node.operand)
    if isinstance(node,ast.Call):
        return (isinstance(node.func,ast.Name) and node.func.id in CONSTRUCTORS and len(node.args)+len(node.keywords)<=8
            and all(closed(x) for x in node.args) and all(k.arg is not None and not k.arg.startswith('_') and closed(k.value) for k in node.keywords))
    return False


def observation_expression(function,source):
    if not isinstance(source,str) or len(source)>512:raise ValueError('Bounded closed call required')
    call=ast.parse(source,mode='eval').body
    if (not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!=function
            or len(call.args)+len(call.keywords)>8 or not all(closed(a) for a in call.args)
            or not all(k.arg is not None and closed(k.value) for k in call.keywords)
            or function in {'_probe_args','_probe_kwargs'}):raise ValueError('Only bounded original-entry calls and literal/constructor arguments allowed')
    for node in [*call.args,*[k.value for k in call.keywords]]:
        if any(isinstance(n,ast.Call) and n.func.id==function for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)):
            raise ValueError('Argument constructors must not resolve to the source entry function')
    args='('+', '.join(ast.unparse(a) for a in call.args)+(',' if len(call.args)==1 else '')+')'
    keywords='{'+', '.join(repr(k.arg)+': '+ast.unparse(k.value) for k in call.keywords)+'}'
    return '(lambda _probe_args, _probe_kwargs: ('+function+'(*_probe_args, **_probe_kwargs), _probe_args, _probe_kwargs))('+args+', '+keywords+')\n'


def proposals(candidate,maximum=16):
    if type(maximum) is not int or not 1<=maximum<=16:raise ValueError('Explicit bounded16-probe search required')
    examples=contract(candidate);aliases={};seeds=[]
    class Expand(ast.NodeTransformer):
        def visit_Name(self,node):return copy.deepcopy(aliases[node.id]) if node.id in aliases else node
    for example in examples:
        try:tree=ast.parse(example.source)
        except SyntaxError:continue
        for statement in tree.body:
            if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):
                value=Expand().visit(copy.deepcopy(statement.value))
                if closed(value):aliases[statement.targets[0].id]=value
            for node in ast.walk(statement):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id==candidate['function']:
                    call=Expand().visit(copy.deepcopy(node))
                    if (all(closed(a) for a in call.args) and all(k.arg is not None and closed(k.value) for k in call.keywords)
                            and len(call.args)+len(call.keywords)<=8):seeds.append(call)
    seen={ast.unparse(s) for s in seeds};result=[]
    # Round-robin seeds and argument slots so one long example does not own
    # the complete bounded budget. Each perturbation changes exactly one slot.
    for value in POOL:
        replacement=ast.parse(value,mode='eval').body
        for seed in seeds:
            for kind,index in [('arg',i) for i in range(len(seed.args))]+[('kw',i) for i in range(len(seed.keywords))]:
                call=copy.deepcopy(seed)
                if kind=='arg':call.args[index]=replacement
                else:call.keywords[index].value=replacement
                source=ast.unparse(call)
                if source in seen or len(source)>512:continue
                seen.add(source)
                # Function result and post-call argument state are observed in
                # one expression, including mutations of list/dict arguments.
                try:expression=observation_expression(candidate['function'],source)
                except ValueError:continue
                result.append(dict(call_expression=source,observation_expression=expression,changed_slot=[kind,index],replacement_literal=value,
                    origin='bounded_machine_input_perturbation',reference_accepted_execution_not_formal_input_domain_certificate=True))
                if len(result)==maximum:return result
    return result


def execute_suite(candidate,probes,*,program=None,origin=None,seconds=5):
    examples=contract(candidate)
    if program is None:program=candidate['program'];origin='public_reference'
    if origin not in ('public_reference','machine_noop','own_policy','test_fixture'):raise ValueError('Explicit actual program origin required')
    if not isinstance(program,str) or not 0<len(program.encode())<=65536 or not 0<seconds<=30:raise ValueError('Bounded actual complete program required')
    tree=ast.parse(program)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!=candidate['function'] or unresolved_globals(program):
        raise ValueError('One complete entry function and declared stdlib context required')
    if len(probes)>16 or any(not isinstance(p.get('observation_expression'),str) or len(p['observation_expression'])>2048 for p in probes):
        raise ValueError('At most16explicit bounded input observations required')
    if any(p['observation_expression']!=observation_expression(candidate['function'],p['call_expression']) for p in probes):
        raise ValueError('Every exact closed call/post-call-state observation must reconstruct')
    commands=[e.source for e in examples]+[p['observation_expression'] for p in probes]
    with tempfile.TemporaryDirectory(prefix='slm-public-train-probes-') as tmp:
        work=Path(tmp);(work/'program.py').write_text(program);(work/'prelude.py').write_text(PRELUDE);(work/'harness.py').write_bytes(HARNESS.read_bytes())
        settings=dict(program=str(work/'program.py'),prelude=str(work/'prelude.py'),function=candidate['function'],examples=commands)
        (work/'settings.json').write_text(json.dumps(settings));policy=python_policy(work)
        policy['command'][-1]=str(work/'harness.py');policy['command'].append(str(work/'settings.json'));policy['memory_bytes']=512*1024**2;policy['cpu_seconds']=max(1,math.ceil(seconds))
        raw=run_bounded(policy,b'',seconds=seconds,output_bytes=1024**2,python=True)
    status={k:v for k,v in raw.items() if k not in ('stdout','stderr')};status['stderr']=raw['stderr'][-2000:].decode(errors='replace')
    receipt=dict(source_id=candidate['source']['id'],actual_program_sha256=sha(program),original_published_commands=len(examples),
        added_machine_probe_commands=len(probes),original_published_commands_unchanged=True,expected_outputs_not_sent_to_worker=True,
        executed_program_origin=origin,executed_program_matches_original_bytes=program==candidate['program'],
        only_one_program_sent_to_worker=True,execution=status,complete=False)
    if raw['status']!='completed':return receipt
    try:
        payload=json.loads(raw['stdout']);rows=payload['results']
        if (set(payload)!={'format','results'} or payload['format']!='own-documented-doctest-raw-output-v1' or len(rows)!=len(commands)
                or any(set(r)!={'stdout','exception_only','entry_body_calls'} or not isinstance(r['stdout'],str)
                    or type(r['entry_body_calls']) is not int or r['entry_body_calls']<0
                    or r['exception_only'] is not None and not isinstance(r['exception_only'],str) for r in rows)):return receipt
    except (ValueError,KeyError,TypeError):return receipt
    return dict(receipt,complete=True,raw_results=rows,original_example_passes=[compare(e,r) for e,r in zip(examples,rows)])


def qualify(candidate,maximum=16,seconds=5):
    proposed=proposals(candidate,maximum);base=dict(source_id=candidate['source']['id'],original_program_sha256=candidate['program_sha256'],
        proposed=proposed,reference_program_not_a_model_teacher=True,machine_labels_not_new_human_judgments=True,
        finite_reference_agreement_not_general_correctness=True,not_a_formal_input_domain_certificate=True,
        original_published_examples_unchanged=True,training_probes_only_no_held_benchmark_used=True)
    if not proposed:return dict(base,available=False,status='no_eligible_closed_original_call_argument_sites',accepted=[])
    reference=execute_suite(candidate,proposed,seconds=seconds)
    if not reference['complete'] or not all(reference['original_example_passes']):
        return dict(base,available=False,status='bounded_reference_discovery_incomplete_or_original_context_failed',accepted=[],raw_reference_discovery=reference)
    try:noop=pass_ablation(candidate)
    except ValueError as error:return dict(base,available=False,status='source_body_ablation_unavailable',accepted=[],qualification=str(error),raw_reference_discovery=reference)
    control=execute_suite(candidate,proposed,program=noop,origin='machine_noop',seconds=seconds)
    if not control['complete']:
        return dict(base,available=False,status='bounded_noop_discovery_incomplete',accepted=[],raw_reference_discovery=reference,raw_noop_discovery=control)
    n=reference['original_published_commands'];accepted=[];dispositions=[]
    for i,probe in enumerate(proposed):
        row=reference['raw_results'][n+i];baseline=control['raw_results'][n+i];reason=None
        if row['exception_only'] is not None:reason='reference_did_not_accept_this_input'
        elif row['entry_body_calls']==0:reason='original_entry_body_not_observed'
        elif len(row['stdout'])>8192:reason='reference_observation_above_literal_budget'
        else:
            try:ast.literal_eval(row['stdout'].strip())
            except (ValueError,SyntaxError,MemoryError,RecursionError):reason='nonliteral_or_unstable_reference_observation'
        if reason is None and row['stdout']==baseline['stdout'] and baseline['exception_only'] is None:
            reason='no_observed_entry_body_effect_against_noop'
        if reason is None:
            accepted.append(dict(probe,expected_observation_stdout=row['stdout'],observed_noop_stdout=baseline['stdout'],
                observed_noop_exception=baseline['exception_only'],proposal_position=i))
        dispositions.append(dict(proposal_position=i,status=reason or 'accepted_literal_body_dependent_reference_observation'))
    return dict(base,available=bool(accepted),status='bounded_reference_probes_prepared' if accepted else 'no_informative_accepted_literal_observations',
        accepted=accepted,dispositions=dispositions,no_op_program_sha256=sha(noop),raw_reference_discovery=reference,raw_noop_discovery=control)


def evaluate(candidate,reference_observation,causal_control,qualification,response,*,origin='own_policy',seconds=5):
    from .causal_code_reward_v2 import evaluate as original_reward
    contract(candidate)
    if (origin not in ('own_policy','test_fixture') or qualification.get('source_id')!=candidate['source']['id']
            or qualification.get('original_program_sha256')!=candidate['program_sha256']):raise ValueError('Exact original train source/probe identity and actual response origin required')
    if not qualification.get('available'):
        return dict(original_reward(candidate,reference_observation,causal_control,response,origin=origin,seconds=seconds),
            augmented_probe_status='unavailable_original_published_reward_only',new_probe_count=0)
    accepted=qualification['accepted'];expected=qualification['raw_reference_discovery'];control=qualification['raw_noop_discovery']
    examples=contract(candidate);n=len(examples)
    if (reference_observation.get('source_id')!=candidate['source']['id'] or reference_observation.get('program_sha256')!=candidate['program_sha256']
            or reference_observation.get('all_original_examples_passed') is not True
            or causal_control.get('source_id')!=candidate['source']['id'] or causal_control.get('original_program_sha256')!=candidate['program_sha256']
            or expected.get('source_id')!=candidate['source']['id'] or control.get('source_id')!=candidate['source']['id']
            or expected.get('actual_program_sha256')!=candidate['program_sha256'] or expected.get('executed_program_origin')!='public_reference'
            or control.get('actual_program_sha256')!=sha(pass_ablation(candidate)) or control.get('executed_program_origin')!='machine_noop'
            or expected.get('original_published_commands')!=n or control.get('original_published_commands')!=n
            or expected.get('added_machine_probe_commands')!=len(qualification['proposed']) or control.get('added_machine_probe_commands')!=len(qualification['proposed'])):
        raise ValueError('Exact original train source, verified reference and actual counterfactual/probe provenance required')
    if (not expected['complete'] or not all(expected['original_example_passes']) or not control['complete']
            or qualification['no_op_program_sha256']!=sha(pass_ablation(candidate))):raise ValueError('Actual complete reference/noop discovery identity required')
    if ([compare(e,r) for e,r in zip(examples,expected['raw_results'][:n])]!=[True]*n
            or len(expected['raw_results'])!=n+len(qualification['proposed']) or len(control['raw_results'])!=len(expected['raw_results'])):
        raise ValueError('Every actual original reference output and proposed observation must reconstruct')
    for p in accepted:
        i=p['proposal_position'];ref=expected['raw_results'][n+i];noop=control['raw_results'][n+i]
        if (p['observation_expression']!=qualification['proposed'][i]['observation_expression']
                or p['expected_observation_stdout']!=ref['stdout'] or ref['exception_only'] is not None or ref['entry_body_calls']==0
                or noop['exception_only'] is None and noop['stdout']==ref['stdout']):raise ValueError('Every added actual body-dependent reference observation must reconstruct')
    receipt=dict(format='own-bounded-train-reference-probe-reward-v1',source_id=candidate['source']['id'],candidate_origin=origin,
        original_program_sha256=candidate['program_sha256'],candidate_response_sha256=sha(response),reward=0.,new_probe_count=len(accepted),
        expected_outputs_and_separate_reference_implementation_not_sent_to_candidate_worker=True,
        machine_generated_training_probes_not_publisher_examples=True,genuine_human_preference=False,
        reference_agreement_not_formal_domain_or_general_correctness=True,goal_achieved=False)
    try:actual=execute_suite(candidate,accepted,program=instruction_code(response),origin=origin,seconds=seconds)
    except (ValueError,SyntaxError) as error:return dict(receipt,status='candidate_context_or_function_contract_failure',qualification=str(error))
    if not actual['complete']:return dict(receipt,status='incomplete_candidate_execution',raw_candidate_execution=actual)
    original=actual['original_example_passes'];extra=[r['exception_only'] is None and r['stdout']==p['expected_observation_stdout'] for r,p in zip(actual['raw_results'][n:],accepted)]
    # Original causal cells include post-call checks; setup cells cannot earn
    # partial credit. New qualified observations all distinguish the no-op.
    original_indices=causal_control.get('dependent_example_indices',[]) if causal_control['supported'] else []
    if causal_control['supported']:
        base=causal_control['counterfactual']['raw_machine_mutant_execution'];flags=[compare(e,r) for e,r in zip(examples,base['raw_results'])]
        if len(flags)!=n or original_indices!=[i for i,passed in enumerate(flags) if not passed]:raise ValueError('Every old causal original-example index must reconstruct')
    flags=[original[i] for i in original_indices]+extra;full=all(original+extra);calls=sum(r['entry_body_calls'] for r in actual['raw_results'])
    reward=(1. if full else .9*sum(flags)/len(flags)) if calls else 0.
    return dict(receipt,reward=reward,status='complete_original_and_added_probe_stream_scored',
        all_original_published_examples_passed=all(original),all_augmented_reference_observations_passed=all(extra),
        full_reference_stream_passed=full,added_probe_passes=extra,scored_observations=len(flags),scored_passes=sum(flags),
        raw_candidate_execution=actual)
