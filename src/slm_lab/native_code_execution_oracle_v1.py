"""Bounded public training rewards for exact source-format/native delivery.

Plain Python stays byte exact. A documented single Python fence is a declared
source envelope, removed once without searching for code elsewhere. Original
scoring functions run with an explicitly bound identity parser in private
globals; no existing module, worker or source scorer is monkeypatched.
"""
import copy
import hashlib
import math
from pathlib import Path
from types import FunctionType

from .apps_execution_reward_v1 import reward as apps_reward
from .apps_published_cases_v2 import contract
from .bounded_code_probes_v1 import evaluate as augmented_reward
from .causal_code_reward_v2 import evaluate as causal_reward
from .code_quality import instruction_code
from .corpus import file_hash
from .expanded_execution_corpus_v1 import ExpandedExecutionCorpus

FORMAT = 'own-exact-source-envelope-native-public-code-execution-v1'


def sha(text): return hashlib.sha256(text.encode()).hexdigest()


def identity(program): return program


def raw_function(parent):
    if parent.__globals__.get('instruction_code') is not instruction_code:
        raise ValueError('The original frozen public scorer/parser binding changed')
    environment = dict(parent.__globals__, instruction_code=identity)
    result = FunctionType(parent.__code__, environment, 'native_' + parent.__name__, parent.__defaults__, parent.__closure__)
    result.__kwdefaults__ = copy.deepcopy(parent.__kwdefaults__)
    return result


def safe_observation(value):
    if type(value) is float and not math.isfinite(value):
        return dict(observed_nonfinite_float='NaN' if math.isnan(value) else '+Infinity' if value > 0 else '-Infinity')
    if isinstance(value, dict): return {k: safe_observation(v) for k, v in value.items()}
    if isinstance(value, list): return [safe_observation(v) for v in value]
    if isinstance(value, tuple): return tuple(safe_observation(v) for v in value)
    return value


class NativeCodeExecutionOracle:
    def __init__(self, corpus):
        if (type(corpus) is not ExpandedExecutionCorpus or corpus.role != 'train'
                or corpus.purpose != 'execution_reward_and_complete_reference_CE'):
            raise ValueError('Only the exact admitted complete public human train corpus is allowed')
        self.corpus = corpus
        self.parents = (apps_reward, augmented_reward, causal_reward)
        self.functions = tuple(raw_function(f) for f in self.parents)
        self.initial = tuple((id(f), id(f.__code__), id(f.__globals__['instruction_code'])) for f in self.functions)
        self.provenance = copy.deepcopy(corpus.provenance)
        modules = [Path(__file__), *[Path(f.__code__.co_filename) for f in self.parents]]
        self.sources = {str(p.resolve()): file_hash(p) for p in modules}
        self.assert_bound()

    def assert_bound(self):
        if (self.corpus.role != 'train' or self.corpus.purpose != 'execution_reward_and_complete_reference_CE'
                or self.corpus.provenance != self.provenance
                or any(f.__globals__.get('instruction_code') is not instruction_code for f in self.parents)
                or tuple((id(f), id(f.__code__), id(f.__globals__.get('instruction_code'))) for f in self.functions) != self.initial
                or any(f.__globals__.get('instruction_code') is not identity for f in self.functions)):
            raise ValueError('Immutable complete train source and explicitly separate raw scorer bindings required')
        self.corpus.guard.check()

    def binding(self):
        self.assert_bound()
        return dict(format=FORMAT, original_source_provenance=copy.deepcopy(self.provenance),
            source_files_sha256=dict(self.sources), declared_source_formats=['plain_Python', 'one_exact_Python_fence'],
            original_parent_scorer_globals_unchanged=True, code_extraction_search_disabled=True,
            native_nonterminal_special_tokens_rejected=True, true_generated_EOS_required_for_full_credit=True,
            public_finite_machine_feedback_not_human_or_general_correctness=True, production_RL_admitted=False)

    def source_information(self, position):
        self.assert_bound()
        row = self.corpus.row(position)
        text = row['original']['response']
        if row['kind'] == 'documented':
            candidate = self.corpus.documented_reward.candidates[row['source_id']]
            expected = candidate['program_sha256']
        else:
            expected = row['original']['original_source_body_sha256']
        if sha(text) == expected:
            kind, program = 'plain_Python', text
        elif text.startswith('```python\n') and text.endswith('\n```') and sha(text[len('```python\n'):-len('\n```')]) == expected:
            kind, program = 'one_exact_Python_fence', text[len('```python\n'):-len('\n```')]
        else:
            raise ValueError('No declared byte-exact source envelope matches the original retained human program')
        ids, labels = self.corpus.sequence(position)
        start = row['prefix_positions']
        if (ids[-1] != 2 or labels != [-100] * start + ids[start:]
                or self.corpus.tokenizer.decode(ids[start:-1], skip_special_tokens=False) != text
                or row['response_EOS_targets'] != len(ids) - start):
            raise ValueError('Every complete original source/mask/answer/EOS byte must reconstruct')
        return dict(source_position=position, source_id=row['id'], kind=row['kind'], source_format=kind,
            original_complete_response_sha256=sha(text), original_execution_program_sha256=sha(program),
            complete_original_context_tokens=len(ids), complete_original_response_EOS_targets=len(ids) - start,
            full_unhinted_original_action_budget=row['full_training_action_budget'],
            original_human_reference_is_not_a_generated_sample=True)

    def render(self, position, response):
        info = self.source_information(position)
        if not isinstance(response, str): raise ValueError('The complete actual decoded native response is required')
        if info['source_format'] == 'one_exact_Python_fence':
            if not response.startswith('```python\n') or not response.endswith('\n```'):
                return dict(valid=False, status='declared_complete_Python_fence_not_delivered', actual_response=response,
                    actual_response_sha256=sha(response), source=info)
            program = response[len('```python\n'):-len('\n```')]
        else:
            program = response
        return dict(valid=True, status='declared_source_envelope_decoded_without_code_search', source=info,
            actual_response=response, actual_response_sha256=sha(response), exact_execution_program=program,
            exact_execution_program_sha256=sha(program), code_bytes_inside_declared_envelope_unchanged=True)

    def score_response(self, position, response, *, stopped_on_EOS, origin):
        if type(stopped_on_EOS) is not bool or origin not in ['own_policy', 'original_reference_control', 'test_fixture']:
            raise ValueError('Actual native EOS termination and response origin must be explicit')
        rendered = self.render(position, response)
        receipt = dict(format=FORMAT, rendered_response=rendered, actual_delivered_EOS=stopped_on_EOS, actual_response_origin=origin,
            human_preferences=False, unassisted_capability_score=False, general_correctness_or_efficiency_proven=False)
        if not rendered['valid']:
            return dict(receipt, reward=0., status=rendered['status'], actual_original_oracle_observation=None)
        row = self.corpus.row(position)
        program = rendered['exact_execution_program']
        if row['kind'] == 'apps':
            path = self.corpus.root / 'data/public-apps-original-inventory-v1/original-train' / row['original']['source_problem_id'] / 'input_output.json'
            cases = contract(path.read_bytes())
            actual = self.functions[0](row['original'], cases, self.corpus.apps_controls[row['source_id']],
                program, stopped_on_eos=stopped_on_EOS)
        else:
            catalog = self.corpus.documented_reward
            candidate, reference = catalog.candidates[row['source_id']], catalog.references[row['source_id']]
            control = catalog.original.controls[row['source_id']]
            probes = catalog.rows[row['source_id']]
            parent_origin = 'own_policy' if origin == 'own_policy' else 'test_fixture'
            actual = (self.functions[1](candidate, reference, control, probes['qualification'], program, origin=parent_origin)
                if probes['qualified_for_independent_reader'] else self.functions[2](candidate, reference, control, program, origin=parent_origin))
        observed = actual.get('actual_execution', actual.get('raw_candidate_execution'))
        if observed is not None:
            actual_hash = observed.get('actual_program_sha256', observed.get('original_program_sha256', observed.get('program_sha256')))
            if actual_hash != rendered['exact_execution_program_sha256']:
                raise ValueError('The actual worker program bytes differ from the complete declared native program')
        original = actual.get('reward')
        informative = actual.get('informative') is not False and actual.get('supported') is not False
        if type(original) not in [int, float] or not math.isfinite(original) or not 0 <= original <= 1:
            raise ValueError('An explicit finite bounded actual oracle reward is required')
        if original > 0 and observed is None:
            raise ValueError('Positive reward requires retained actual worker program/output observations')
        result = dict(receipt, actual_original_oracle_observation=safe_observation(actual), original_oracle_reward=original,
            reward=(min(original, .9) if not stopped_on_EOS else original) if informative else None,
            status=actual.get('status'), source_reward_informative=informative,
            all_observed_worker_program_bytes_equal_complete_native_program=observed is not None)
        self.assert_bound()
        return result

    def delivery(self, position, public_reference_prefix_ids, actions, *, delivered_action_budget, origin):
        self.source_information(position)
        if (not isinstance(public_reference_prefix_ids, list) or not isinstance(actions, list)
                or type(delivered_action_budget) is not int or not 1 <= len(actions) <= delivered_action_budget
                or origin not in ['own_policy', 'original_reference_control', 'test_fixture']
                or any(type(t) is not int or not 0 <= t < 32768 for t in public_reference_prefix_ids + actions)
                or 2 in actions[:-1] or actions[-1] != 2 and len(actions) != delivered_action_budget):
            raise ValueError('Every actual native action requires first EOS or its complete declared horizon')
        row = self.corpus.row(position)
        ids, _ = self.corpus.sequence(position)
        if public_reference_prefix_ids != ids[row['prefix_positions']:row['prefix_positions'] + len(public_reference_prefix_ids)]:
            raise ValueError('Supplied training code must be an exact prefix of the original human response')
        stopped = actions[-1] == 2
        body = actions[:-1] if stopped else actions
        response = self.corpus.tokenizer.decode(public_reference_prefix_ids + body, skip_special_tokens=False)
        raw = self.corpus.tokenizer.decode(public_reference_prefix_ids + actions, skip_special_tokens=False)
        native = dict(actual_generated_suffix_ids=list(actions), actual_supplied_public_reference_ids=list(public_reference_prefix_ids),
            actual_response_with_native_control_tokens_preserved=raw, complete_actual_response=response,
            actual_response_origin=origin, source_episode_qualification_still_required=True,
            actual_stopped_on_EOS=stopped, actual_delivered_action_budget=delivered_action_budget,
            terminal_EOS_is_a_declared_boundary_not_executed_Python=True)
        if any(t in [0, 1, 3] for t in actions):
            return dict(native, format=FORMAT, reward=0., status='invalid_generated_native_control_token',
                actual_original_oracle_observation=None, all_generated_tokens_preserved_without_resampling=True)
        return dict(self.score_response(position, response, stopped_on_EOS=stopped, origin=origin), **native)
