import copy
import pytest
from slm_lab.clevr_scene_supervision import description


def scene():
    return {'objects':[{'size':'large','color':'red','material':'metal','shape':'cube','pixel_coords':[300,100,10]},
                       {'size':'small','color':'blue','material':'rubber','shape':'sphere','pixel_coords':[100,100,10]},
                       {'size':'large','color':'green','material':'metal','shape':'cube','pixel_coords':[200,100,10]}]}


def test_dense_targets_use_projected_order_and_exact_source_object_counts():
    result=description(scene())
    assert result.splitlines()[1]=='1. small blue rubber sphere.'
    assert result.endswith('Shape counts: cubes 2, cylinders 0, spheres 1.')


@pytest.mark.parametrize('change',['attribute','outside','nonfinite','count'])
def test_invalid_source_objects_are_not_silently_invented_or_clipped(change):
    value=copy.deepcopy(scene())
    if change=='attribute':value['objects'][0]['shape']='triangle'
    elif change=='outside':value['objects'][0]['pixel_coords'][0]=500
    elif change=='nonfinite':value['objects'][0]['pixel_coords'][0]=float('nan')
    else:value['objects'].pop()
    with pytest.raises(ValueError):description(value)
