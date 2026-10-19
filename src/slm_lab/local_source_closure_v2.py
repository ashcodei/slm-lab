"""Resolve reachable local Python imports and nested immutable evidence bindings."""
import ast
import json
from pathlib import Path


def python_sources(root, seeds):
    root = Path(root).resolve()
    bases = [root/'src', root/'scripts', root/'tests']
    found = set(); pending = [Path(p).resolve() for p in seeds]
    def resolve(name):
        result = []
        for base in bases:
            path = base.joinpath(*name.split('.'))
            for candidate in [path.with_suffix('.py'), path/'__init__.py']:
                if candidate.is_file(): result.append(candidate.resolve())
        return result
    while pending:
        path = pending.pop()
        if path in found: continue
        if path.suffix != '.py' or not path.is_file(): raise ValueError('Existing local Python source required')
        found.add(path)
        relative = next((path.relative_to(base) for base in bases if path.is_relative_to(base)), None)
        if relative is None: raise ValueError('Local source outside declared repository roots')
        parts = list(relative.with_suffix('').parts)
        package = parts[:-1]
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names: pending.extend(resolve(alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package) + 1: raise ValueError('Relative import escapes declared local package')
                    prefix = package[:len(package)-node.level+1]
                    target = '.'.join(prefix + ([node.module] if node.module else []))
                else: target = node.module or ''
                if target: pending.extend(resolve(target))
                for alias in node.names:
                    if alias.name != '*': pending.extend(resolve('.'.join(p for p in [target,alias.name] if p)))
    return sorted(found)


def evidence_bindings(initial):
    """Expand every referenced JSON source map without changing its expected SHA."""
    from .corpus import file_hash
    result = dict(initial); pending = list(result); seen = set()
    while pending:
        name = pending.pop()
        if name in seen: continue
        seen.add(name); path = Path(name)
        if path.suffix != '.json': continue
        if file_hash(path) != result[name]: raise ValueError('Existing evidence SHA differs before source-map expansion')
        raw = path.read_bytes()
        if b'"source_files_sha256"' not in raw: continue
        # Counts/case values are irrelevant to the SHA map. Preserve arbitrary
        # JSON integers as text without changing Python's process-wide limit.
        value = json.loads(raw,parse_int=str)
        if not isinstance(value, dict): continue
        mapping = value.get('source_files_sha256', {})
        if not isinstance(mapping, dict): raise ValueError('Evidence source binding must be an explicit mapping')
        for child, digest in mapping.items():
            # Evidence may store repository-relative paths. Resolve these
            # against the caller's repository cwd, never against the report.
            child = str(Path(child).resolve())
            if child in result and result[child] != digest: raise ValueError('Conflicting immutable source bindings')
            if child not in result: result[child] = digest; pending.append(child)
    return result
