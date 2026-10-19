"""Explicit train-only source qualification for complete tree-building helpers.

The original Record/TreeNode starters are supplied unchanged. TreeNode differs
from the archived reference in access modifiers. Complete published tests must
verify this exact supplied input; no held-evaluation runner policy is changed.
"""
import hashlib

from .exercise_java_runner import PublishedSuites


class TrainingSuites(PublishedSuites):
    def layout(self,row):
        if row['id']!='exercism:java:tree-building':return super().layout(row)
        if row['split']!='train' or row['group_id']!='exercism:tree-building':
            raise ValueError('This reviewed helper qualification applies only to its original training source')
        base,config=self.config(row)
        expected={'src/main/java/BuildTree.java':'.meta/src/reference/java/BuildTree.java'}
        supplied=['src/main/java/Record.java','src/main/java/TreeNode.java']
        if (set(config['solution'])!=set(expected)|set(supplied)
                or config['example']!=list(expected.values())
                or config['editor']!=['src/main/java/InvalidRecordsException.java']):
            raise ValueError('Exact original reviewed tree-building interface required')
        provided={}
        for name in supplied:
            raw=self.read(base+name)
            item=next(r for r in row['source_files'] if r['path']==base+name)
            if hashlib.sha256(raw).hexdigest()!=item['sha256']:
                raise ValueError('Reviewed supplied helper must match the original publisher starter bytes')
            provided[name]=raw
        return base,config,expected,provided
