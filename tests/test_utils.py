import pytest
from utils import density_matrix, kraus, System, operator
import numpy as np
def test_partial_trace():
    np.random.seed(0)
    a = density_matrix(np.random.rand(2))
    b = density_matrix(np.random.rand(2))
    c = density_matrix(np.random.rand(2))
    rho = a.tensor(b).tensor(c)
    assert b.tensor(c) == rho.partial_trace(0)

def test_kraus_multiplication():
    np.random.seed(0)
    U = kraus(np.random.rand(3,2,2))
    rho = density_matrix(np.random.rand(2))
    assert U * rho == [u * rho for u in U]

@pytest.fixture
def sample_system():
    return System(3), operator([[1,1],[-1,1]]) / np.sqrt(2)
def test_system_transition(sample_system):
    sys, U = sample_system
    sys.transition(U)
    assert sys.rho == density_matrix(3*[[[1/2,-1/2],[-1/2,1/2]]]) 

def test_system_transition_single_qubit(sample_system):
    sys, U = sample_system
    sys.transition_qubit(U,[1])
    assert sys.rho == density_matrix([[[1,0],[0,0]],[[1/2,-1/2],[-1/2,1/2]],[[1,0],[0,0]]])

def test_system_transition_two_qubits(sample_system):
    sys, U = sample_system
    sys.transition_qubit(U.tensor(U),[0,2])
    assert sys.rho == density_matrix([[[1/2,-1/2],[-1/2,1/2]],[[1,0],[0,0]],[[1/2,-1/2],[-1/2,1/2]]])

