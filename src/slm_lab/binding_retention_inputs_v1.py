"""Match each proposed public-data exposure to its complete actual loss mask."""
from .instruction_generalization_trial_v3 import complete_encoding, LANGUAGES


def update_rows(pools, primary, rehearsal, encoder):
    if len(primary) != 4 or primary[2]['pool'] != 'verified_python' or primary[3]['pool'] != 'human_instruction':
        raise ValueError('Two modules, one verified program and one human demonstration required')
    if primary[0]['pool'] not in LANGUAGES or primary[1]['pool'] not in LANGUAGES or primary[0]['pool'] == primary[1]['pool']:
        raise ValueError('Two distinct declared module languages required')
    if len({r['update'] for r in [*primary, rehearsal]}) != 1:
        raise ValueError('Exposure records refer to different updates')
    result = []
    for pick in [*primary, {**rehearsal, 'pool': 'retention'}]:
        name, index = pick['pool'], pick['index']
        if type(index) is not int or not 0 <= index < len(pools[name]):
            raise ValueError('Recorded source index is invalid')
        row = pools[name][index]
        if row['id'] != pick['id'] or row.get('image'):
            raise ValueError('Recorded original text-only source differs')
        actual = complete_encoding(row, encoder)
        if any(actual[k] != pick[k] for k in ['target_tokens', 'context_tokens']):
            raise ValueError('Recorded complete target/mask budget differs')
        result.append(row)
    return result
