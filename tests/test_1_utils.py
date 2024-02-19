from gammabayes.utils import update_with_defaults, haversine


def test_update_with_defaults():
    test_default    = {'a':1, 'b':2, 'c':3, 'd':4}
    test_target     = {'a':[1,2,3,4], 'c':'This is a test ya c'}

    update_with_defaults(target_dict=test_target, default_dict=test_default)

    assert test_target == {'a':[1,2,3,4], 'b':2, 'c':'This is a test ya c', 'd':4}

def test_haversine():
    lon1, lat1, lon2, lat2 = 0,0,0,0
    assert haversine(lon1, lat1, lon2, lat2) == 0

    lon1, lat1, lon2, lat2 = 180,0,0,0
    assert haversine(lon1, lat1, lon2, lat2) == 180