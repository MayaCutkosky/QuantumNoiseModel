import pytest
from models1 import Model, load_model
from qiskit_ibm_runtime.fake_provider import FakeFez


@pytest.fixture
def model():
    prop = FakeFez().properties()
    return Model(prop)

def check_cross_talk(model):
    np.sum(model.cross_talk_probabilities

def test_run(model):
    circuit =  [('x', (0,), None),('x', (1,), None)  ]
    prob = model.run(circ, 4)
    assert prob.shape == (4,2)
    assert [abs(p[0] + p[1] - 1) < 1e-10 for p in prob]
    assert p[0][1] > 0.9
    assert p[1][1] > 0.9
    assert p[2][0] > 0.9

def test_save(model):
    model.save()
    reloaded_model = load_model()
    assert np.sum(model.run() == reloaded_model.run())
