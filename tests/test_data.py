import pytest
from data import process_backend, relaxation_error, gate_error, process_gate_data, get_errs_from_gate
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit_ibm_runtime.models.backend_properties import BackendProperties
from numpy import ndarray
import pickle
from qiskit.quantum_info import Kraus as qiskit_Kraus
from utils import Kraus
import numpy as np
@pytest.fixture
def prop():
    backend = FakeFez() 
    return backend.properties()

gates = list(FakeFez().properties().gates)
standard_gates = []
for g in gates[:10]:
    if g.gate in ['cz', 'sx', 'rz', 'id', 'x']:
        standard_gates.append(g)


@pytest.mark.parametrize('gate',  standard_gates)
@pytest.mark.dependency(name = 'test_process_gate_data')
def test_process_gate_data(gate):
    name, qubits, gate_err, gate_length = process_gate_data(gate)
    assert isinstance(qubits, tuple)
    assert len(qubits) in [1,2]
    assert gate_err <= 1
    assert gate_err >= 0
    assert isinstance(gate_length, float)
    assert gate_length >= 0
    assert gate_length < 1e-5
    
        

@pytest.mark.parametrize('gate', standard_gates )
@pytest.mark.dependency(name = 'relaxation_error', depends=["Kraus", 'test_process_gate_data'],scope='session')
def test_relaxation_error(prop, gate):
    name, qubits, gate_err, gate_length = process_gate_data(gate)
    relax_error = relaxation_error(prop, qubits[0], gate_length)
    assert np.isfinite(relax_error).prod() == 1
    if name in ['cz', 'x']:
        assert len(relax_error) == 3
    for K in relax_error:
        assert K.shape == (2,2)


from jaxlib._jax import ArrayImpl
@pytest.mark.parametrize('gate', standard_gates )
@pytest.mark.dependency(name = 'get_all_error', depends = ['relaxation_error'], scope = 'module')
def test_all_error(prop,gate):
    r_err, g_err = get_errs_from_gate(prop, gate)
    for err in [r_err, g_err]:
        assert isinstance(err, Kraus)
        assert isinstance(err._data, np.ndarray) or isinstance(r_err, ArrayImpl)
        assert err.is_Kraus()
    assert np.isfinite(g_err._data).prod() == 1
    

@pytest.mark.dependency()
def test_process_backend_datatypes(prop):
    output = process_backend(prop)
    assert isinstance(output[0], int)
    assert isinstance(output[1], int)
    assert isinstance(output[3], dict)
    assert isinstance(output[4], ndarray)

