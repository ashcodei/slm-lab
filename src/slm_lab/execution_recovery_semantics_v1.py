"""Compare executed code records without treating wall time as model state.

Every action, log probability, reward and observed output is retained. Only
the OS-jail execution wrapper's measured elapsed seconds may differ on replay.
Raw records remain intact in the original journals and checkpoint payloads.
"""


def semantic_record(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == 'execution' and isinstance(item, dict):
                result[key] = {k:semantic_record(v) for k,v in item.items() if k != 'seconds'}
            else:
                result[key] = semantic_record(item)
        return result
    if isinstance(value, list):
        return [semantic_record(item) for item in value]
    if isinstance(value, tuple):
        return tuple(semantic_record(item) for item in value)
    return value


def same_executed_update(left, right):
    return semantic_record(left) == semantic_record(right)
