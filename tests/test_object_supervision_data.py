from slm_lab.object_supervision_data import acceptable


def box(label='first', region=(.1, .1, .5, .5), **flags):
    return {'label_id': label, 'box': list(region), 'source': 'xclick',
            'flags': {name: flags.get(name, 0) for name in ['IsGroupOf', 'IsDepiction', 'IsInside', 'IsOccluded', 'IsTruncated']}}


def test_object_supervision_requires_clear_unambiguous_original_human_box():
    first = box()
    assert acceptable(first, [first, box('second', (.6, .6, .9, .9))])
    assert not acceptable(first, [first, box('second')])
    assert not acceptable(box(IsGroupOf=1), [])
    assert not acceptable(box(IsOccluded=1), [])
    assert not acceptable({**first, 'source': 'machine'}, [])
