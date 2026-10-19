"""Train-fitted constant and literal input-copy controls, never neural scores.

These provide a low-effort reference for future source-reserved evaluation.
Prediction accepts only task and original question; reserved answers cannot
enter the predictor. Selection between copying and the training majority is
fixed using training accuracy alone, with ties going to the majority.
"""
from collections import Counter
import re


def copy_input(task,question):
    if task not in ('binary_matrix','rotate_matrix'):return None
    values=[]
    for line in reversed(question.rstrip().splitlines()):
        if not re.fullmatch(r'\d+(?: +\d+)*',line):break
        values.append([int(value) for value in line.split()])
    values.reverse()
    if not 1<=len(values)<=30 or any(len(row)!=len(values) for row in values):
        raise ValueError('The complete original square input grid is required')
    return '\n'.join(' '.join(map(str,row)) for row in values)


def fit(train_rows):
    if not train_rows or any(row['split']!='train' for row in train_rows):
        raise ValueError('Only the complete explicit original training role may fit these controls')
    result={}
    for task in sorted({row['task'] for row in train_rows}):
        selected=[row for row in train_rows if row['task']==task]
        labels=Counter(row['response'] for row in selected)
        majority=min(labels,key=lambda value:(-labels[value],value))
        copied=[copy_input(task,row['original_entry']['question']) for row in selected]
        copy_correct=sum(value is not None and value==row['response'] for row,value in zip(selected,copied))
        copy_available=all(value is not None for value in copied)
        choose_copy=copy_available and copy_correct>labels[majority]
        result[task]=dict(train_examples=len(selected),train_distinct_canonical_answers=len(labels),
            train_majority_answer=majority,train_majority_correct=labels[majority],
            copy_input_available=copy_available,train_input_copy_correct=copy_correct if copy_available else None,
            selected_control='copy_input' if choose_copy else 'training_majority',
            selection_used_only_training_labels=True)
    return result


def predict(fitted,task,question,*,method='train_selected'):
    if task not in fitted or method not in ('training_majority','copy_input','train_selected'):
        raise ValueError('Only the bound original task and declared simple prediction control are allowed')
    if method=='train_selected':method=fitted[task]['selected_control']
    if method=='copy_input':
        value=copy_input(task,question)
        if value is None:raise ValueError('Input copying is only defined for the two original matrix-output tasks')
        return value
    return fitted[task]['train_majority_answer']
