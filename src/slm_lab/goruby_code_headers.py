"""Extract documented method interfaces with isolated, pinned syntax grammars."""
import importlib
import json
from pathlib import Path
import sys

from .corpus import file_hash


class HeaderParser:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / '.tools/goruby-parser-v1.json'
        self.manifest_sha = file_hash(self.manifest_path)
        metadata = json.loads(self.manifest_path.read_text())
        target = Path(metadata['toolchain_root']).resolve()
        for name, digest in metadata['installed_files_sha256'].items():
            if file_hash(target / name) != digest: raise ValueError('Pinned syntax grammar file changed: ' + name)
        for name in ('tree_sitter', 'tree_sitter_go', 'tree_sitter_ruby'):
            existing = sys.modules.get(name)
            if existing and not Path(existing.__file__).resolve().is_relative_to(target):
                raise ValueError('Another syntax grammar package is already loaded')
        sys.path.insert(0, str(target))
        tree_sitter = importlib.import_module('tree_sitter')
        self.parsers = {}
        for label, name in [('Go', 'tree_sitter_go'), ('Ruby', 'tree_sitter_ruby')]:
            module = importlib.import_module(name)
            self.parsers[label] = tree_sitter.Parser(tree_sitter.Language(module.language()))
        self.metadata = metadata

    def measure(self, language, code):
        if language not in self.parsers: raise ValueError('Unsupported header grammar')
        raw = code.encode(); old = code.split('{', 1)[0][:500]
        prefix = b'package original_context\n' if language == 'Go' else b''
        tree = self.parsers[language].parse(prefix + raw)
        if tree.root_node.has_error:
            return {'parsed': False, 'diagnostic': 'Source/context has syntax errors under the pinned grammar'}
        types = {'function_declaration', 'method_declaration'} if language == 'Go' else {'method', 'singleton_method'}
        nodes = [node for node in tree.root_node.named_children if node.type in types]
        other = [node for node in tree.root_node.named_children if node.type not in types | {'comment', 'package_clause'}]
        if len(nodes) != 1 or other:
            return {'parsed': False, 'diagnostic': 'Expected one outer original function/method and no additional declarations'}
        node = nodes[0]; offset = len(prefix)
        name = node.child_by_field_name('name'); parameters = node.child_by_field_name('parameters')
        body = node.child_by_field_name('body')
        if language == 'Go':
            if body is None: return {'parsed': False, 'diagnostic': 'Original Go function has no source body'}
            end = body.start_byte - offset
        else:
            if name is None: return {'parsed': False, 'diagnostic': 'Original Ruby method name missing'}
            # The Ruby body starts after optional separators/comments; the
            # interface ends at the actual method name or parameter list.
            end = (parameters.end_byte if parameters else name.end_byte) - offset
            if parameters:
                stack = [parameters]
                while stack:
                    item = stack.pop()
                    if item.type.startswith('heredoc'):
                        return {'parsed': False, 'diagnostic': 'Ruby heredoc defaults need noncontiguous source-header handling'}
                    stack.extend(item.named_children)
        if not 0 < end <= len(raw): raise ValueError('Parsed header byte range escapes original source')
        header = raw[:end].decode(); old_bytes = len(old.encode())
        parameter_cut = parameters is not None and old_bytes < parameters.end_byte - offset
        body_copy = body is not None and old_bytes > body.start_byte - offset
        return {'parsed': True, 'node_type': node.type, 'original_signature': old,
            'complete_header': header, 'header_end_byte': end,
            'header_text_cut': old_bytes < end, 'parameter_text_cut': parameter_cut,
            'source_after_header_in_context': old_bytes > end and bool(raw[end:old_bytes].strip()),
            'method_body_text_in_context': body_copy,
            'needs_correction': old.strip() != header.strip()}
