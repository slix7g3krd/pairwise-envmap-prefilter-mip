import pytest
import graphics.envmap_prefilter_mip as mod

CASES = [
    ("ggx_prefilter", (1,)),
    ("mip_mapping", (1,)),
    ("cube_seam", (1,)),
    ("async_read", (1,)),
    ("hdr_clamp", (1,)),
    ("determinism", (1,)),
    ("roughness_lut", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("face_order", (1,)),
    ("metrics", (1,)),
    ("sample", (1,)),
    ("lod_smooth", (1,)),
    ("energy", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
