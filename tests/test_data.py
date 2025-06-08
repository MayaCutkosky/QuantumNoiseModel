import pytest
from data import process_backend, relaxation_error, gate_error, process_gate_data
from qiskit_ibm_runtime.fake_provider import FakeFez
from numpy import ndarray
import pickle
from qiskit.quantum_info import Kraus as qiskit_Kraus
import numpy as np
@pytest.fixture
def prop():
    backend = FakeFez() 
    return backend.properties()


@pytest.mark.parametrize('i', range(1640) )
@pytest.mark.dependency(depends=["tests/test_utils.py::Kraus"],scope='session')
def test_relaxation_error(prop,i):
    name, qubits, gate_err, gate_length = process_gate_data(prop.gates[i])
    relax_error = relaxation_error(prop, qubits[0], gate_length)
    assert np.isfinite(relax_error).sum() == 0
    if name in ['cz', 'x']:
        assert len(relax_error) == 3
    assert relax_error.is_Kraus()



@pytest.mark.parametrize('i', range(1640) )
@pytest.mark.dependency('test_relaxation_error')
def test_gate_error(prop,i):
    name, qubits, gate_err, gate_length = process_gate_data(prop.gates[i])
    num_qubits = len(qubits)
    with open('tests/relax_'+str(num_qubits)+'qubit.pickle', 'rb') as f:
        relax_err = qiskit_Kraus(pickle.load(f))
    error = gate_error(gate_err, relax_err, num_qubits)
    assert error * (4**num_qubits)
    
    

@pytest.mark.dependency()
def test_process_backend_datatypes(prop):
    output = process_backend(prop)
    assert isinstance(output[0], int)
    assert isinstance(output[1], int)
    assert isinstance(output[3], dict)
    assert isinstance(output[4], ndarray)

