"""Conservative weak-supervision screening, never an RL reward oracle.

Published synthetic tests establish only agreement with those tests. A passing
program can still be semantically wrong. Every selected row needs fresh local
execution and source/native/held-overlap readback before any SFT admission.
"""
import ast
import hashlib
import json
import math


def published_all_pass(row):
    """Check complete provider metadata; do not label skipped code incorrect."""
    try:
        tests=json.loads(row['unit_tests']);statuses=json.loads(row['tests_execution_status'])
        score=row['average_test_score']
    except (KeyError,TypeError,ValueError):return False
    return (isinstance(tests,list) and 1<=len(tests)<=128 and all(isinstance(t,str) for t in tests)
        and isinstance(statuses,list) and len(statuses)==len(tests)
        and all(s=='pass' for s in statuses) and type(score) in {int,float}
        and math.isfinite(score) and score==1)


def body_fingerprint(program):
    """Ignore formatting/comments/docstrings only; source bytes stay unchanged.

This is a conservative duplicate/veto signal, not semantic equivalence.
Different local names or structurally different bugs are not certified.
"""
    tree=ast.parse(program)
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.body:
            first=node.body[0]
            if isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str):
                node.body=node.body[1:]
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def supervision_scope():
    return dict(synthetic_teacher_data=True,human_preferences=False,own_model_on_policy=False,
        weak_supervised_learning_candidate_only=True,independent_semantic_correctness_certified=False,
        use_as_RL_reward_oracle_forbidden=True,training_admitted=False)
