"""Read original CLEVR JSON arrays and verify symbolic scene answers as data."""
import codecs
import json
import re


def array_items(source, key):
    decoder = json.JSONDecoder(); utf8 = codecs.getincrementaldecoder('utf-8')(); buffer = ''; eof = False
    marker = re.compile('"' + re.escape(key) + '"\\s*:\\s*\\[')
    while True:
        block = source.read(1 << 20)
        if not block: raise ValueError('Original metadata array absent')
        buffer += utf8.decode(block)
        match = marker.search(buffer)
        if match: buffer = buffer[match.end():]; break
        if len(buffer) > 1 << 20: raise ValueError('Original metadata header exceeds limit')
    while True:
        buffer = buffer.lstrip()
        if buffer.startswith(','): buffer = buffer[1:].lstrip()
        if buffer.startswith(']'):
            # Consume to EOF so ZipExtFile verifies the original member CRC.
            while source.read(1 << 20): pass
            return
        try: item, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            if eof: raise ValueError('Incomplete original metadata array')
            block = source.read(1 << 20); eof = not block; buffer += utf8.decode(block, final=eof)
            if len(buffer) > 16 << 20: raise ValueError('Original metadata record exceeds limit')
            continue
        if not isinstance(item, dict): raise ValueError('Expected original scene/question objects')
        buffer = buffer[end:]; yield item


def answer(scene, program):
    objects = scene['objects']; outputs = []
    for position, node in enumerate(program):
        indices = node['inputs']
        if any(type(index) is not int or not 0 <= index < position for index in indices): raise ValueError('Invalid original program dependency')
        inputs = [outputs[index] for index in indices]; values = node['value_inputs']; op = node['function']
        if op == 'scene': value = set(range(len(objects)))
        elif op.startswith('filter_'):
            attr = op.removeprefix('filter_'); value = {index for index in inputs[0] if objects[index][attr] == values[0]}
        elif op == 'unique':
            if len(inputs[0]) != 1: raise ValueError('Original question has an ambiguous unique object')
            value = next(iter(inputs[0]))
        elif op == 'relate': value = set(scene['relationships'][values[0]][inputs[0]])
        elif op.startswith('same_'):
            attr = op.removeprefix('same_'); origin = inputs[0]
            value = {index for index, obj in enumerate(objects) if index != origin and obj[attr] == objects[origin][attr]}
        elif op in {'union', 'intersect'}: value = inputs[0] | inputs[1] if op == 'union' else inputs[0] & inputs[1]
        elif op == 'count': value = len(inputs[0])
        elif op == 'exist': value = bool(inputs[0])
        elif op.startswith('query_'): value = objects[inputs[0]][op.removeprefix('query_')]
        elif op.startswith('equal_'): value = inputs[0] == inputs[1]
        elif op == 'less_than': value = inputs[0] < inputs[1]
        elif op == 'greater_than': value = inputs[0] > inputs[1]
        else: raise ValueError('Unsupported original CLEVR program operator: ' + op)
        outputs.append(value)
    if not outputs: raise ValueError('Empty original question program')
    result = outputs[-1]
    return ('yes' if result else 'no') if type(result) is bool else str(result)
