"""Extract unchanged original held question text from each explicit source schema."""


def original_question(benchmark, row):
    if benchmark == 'NQ-open':
        original = row['source_original_question']
        if not isinstance(original,dict) or not isinstance(original.get('question'),str):
            raise ValueError('Original NQ source object and its unchanged question string required')
        value = original['question']
    elif benchmark == 'GSM8K': value = row['question']
    elif benchmark == 'reserved-MBPP': value = row['text']
    else: raise ValueError('Explicit supported original held source required')
    if not isinstance(value,str) or not value.strip() or '\0' in value:
        raise ValueError('Nonempty original question text without boundary sentinel required')
    return value
