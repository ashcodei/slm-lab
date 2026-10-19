"""Locate executable Python tokens without dropping docstrings or true EOS.

AST columns count UTF-8 bytes; Python lexical tokens and this tokenizer's
offsets count Unicode characters. Convert explicitly before assigning spans.
This is a candidate weighting/diagnostic primitive, not a demonstrated gain.
"""
import ast
import io
import math
import token
import tokenize


def regions(program,tokenizer,*,code_weight=4.):
    if not isinstance(program,str) or not program or not math.isfinite(code_weight) or not 1<=code_weight<=16:
        raise ValueError('Whole valid Python source and explicit bounded positive code weight required')
    tree=ast.parse(program);lines=program.splitlines(keepends=True);starts=[];total=0
    for line in lines:starts.append(total);total+=len(line)
    def lexical_offset(position):
        row,column=position
        if row>len(starts):return len(program)
        return starts[row-1]+column
    def ast_offset(row,column):
        return starts[row-1]+len(lines[row-1].encode('utf-8')[:column].decode('utf-8'))
    docs=bytearray(len(program));code=bytearray(len(program))
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.body:
            first=node.body[0]
            if isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str):
                value=first.value;begin=ast_offset(value.lineno,value.col_offset);end=ast_offset(value.end_lineno,value.end_col_offset)
                docs[begin:end]=b'\1'*(end-begin)
    for value in tokenize.generate_tokens(io.StringIO(program).readline):
        if value.type not in {token.NAME,token.NUMBER,token.OP,token.STRING}:continue
        begin=lexical_offset(value.start);end=lexical_offset(value.end)
        if value.type==token.STRING and any(docs[begin:end]):continue
        for i in range(begin,end):
            if not docs[i]:code[i]=1
    encoded=tokenizer.encode(program,add_special_tokens=False)
    if tokenizer.decode(encoded.ids,skip_special_tokens=False)!=program or len(encoded.ids)!=len(encoded.offsets):
        raise ValueError('Exact original code bytes and actual tokenizer offsets required')
    weights=[];code_mass=doc_mass=0.
    for begin,end in encoded.offsets:
        if not 0<=begin<=end<=len(program):raise ValueError('Tokenizer offsets must refer to complete original Unicode characters')
        width=end-begin;fraction=sum(code[begin:end])/width if width else 0.;document=sum(docs[begin:end])/width if width else 0.
        weights.append(1+(code_weight-1)*fraction);code_mass+=fraction;doc_mass+=document
    weights.append(1.)
    return dict(whole_original_response_EOS_ids=[*encoded.ids,2],positive_target_weights=weights,
        response_EOS_targets=len(weights),executable_equivalent_token_mass=code_mass,
        docstring_equivalent_token_mass=doc_mass,other_and_EOS_equivalent_token_mass=len(weights)-code_mass-doc_mass,
        weighted_target_denominator=math.fsum(weights),code_weight=code_weight,
        every_original_target_retained_with_positive_weight=True,true_EOS_weight=1.)


def weighted_CE(logits,labels,weights):
    import torch
    from torch.nn import functional as F
    if (logits.ndim!=3 or labels.shape!=logits.shape[:2] or weights.shape!=labels.shape or weights.requires_grad
            or bool((labels[:,0]!=-100).any())
            or not bool(torch.isfinite(weights).all()) or bool((weights[labels==-100]!=0).any())
            or bool((weights[labels!=-100]<1).any()) or bool((weights>16).any())
            or not bool((labels[:,1:]!=-100).any())):
        raise ValueError('Every and only actual original target needs finite positive weight; prompt/padding weights must be zero')
    losses=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100,reduction='none')
    shifted=weights[:,1:].reshape(-1).to(losses)
    return (losses*shifted).sum(),shifted.sum()
