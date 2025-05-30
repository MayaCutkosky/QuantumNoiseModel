import pytest
import numpy as np
import jax.numpy as jnp
from utils import Operator, density_matrix, kraus, System, operator



@pytest.fixture
def U():
    return operator([[0,1],[-1,0]])



def operator_data_type(U):
    
    assert U.find_data_type() == 'numpy'
    U = operator(jnp.array([[0,1],[-1,0]]))
    assert operator.find_data_type() == 'jax' 

def test_multiplication(U):
    U2 = U * U
    assert U2.shape == (2,2)
    assert U2 == operator(-1 * np.identity(2))

def test_tensor(U):
    U2 = U.tensor(operator(np.identity(2)))
    assert U2.shape == (4,4)
    assert U2 == operator([[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0]])

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



@pytest.mark.parametrize('U1, U_stacked, expected_vals',[
    (
         operator([[1,1],[1,-1]]) / np.sqrt(2),
         operator([[[1,1],[1,-1]]]*3) / np.sqrt(2),
         [
             density_matrix([[[1/2,1/2],[1/2,1/2]]]*3),
             [density_matrix([[[1,0],[0,0]],[[1/2,1/2],[1/2,1/2]],[[1,0],[0,0]]]), density_matrix([[[1,0],[0,0]]] * 3)],
             density_matrix([[[1/2,1/2],[1/2,1/2]],[[1,0],[0,0]],[[1/2,1/2],[1/2,1/2]]]),
             density_matrix([[[1/2,1/2], [1/2,1/2]]]*3)
         ]
    ),
    (
         kraus([[[1,1],[1,-1]], [[0,1],[1,0]], [[1,0],[0,1]]]) / 2,
         kraus([[[[1,1],[1,-1]], [[0,1],[1,0]], [[1,0],[0,1]]]] * 3) / 2,
         [
             density_matrix([[[.5,.25], [.25,.5]]]*3),
             [density_matrix([[[1,0],[0,0]],[[.5,.25], [.25,.5]],[[1,0],[0,0]]]), density_matrix([[[1,0],[0,0]], [[0.625,0.125],[0.125,0.375]],[[1,0],[0,0]]])],
             density_matrix([[[.5,.25], [.25,.5]],[[1,0],[0,0]],[[.5,.25], [.25,.5]]]),
             density_matrix([[[.5,.25], [.25,.5]]]*3)
         ]
    )
])    
class Test_system:

    def test_system_transition_single_qubit(self,U1, U_stacked, expected_vals):
        sys = System(3)
        sys.transition_qubit(U1,[1])
        assert sys.rho == expected_vals[1][0]
        sys.transition_qubit(U1, [1])
        assert sys.rho == expected_vals[1][1]
    
    def test_system_transition_two_qubits(self,U1, U_stacked,  expected_vals):
        sys = System(3)
        sys.transition_qubit(U1.tensor(U1),[0,2])
        assert sys.rho == expected_vals[2]
    
    def test_system_transition(self,U1,U_stacked, expected_vals):
        sys = System(3)
        sys.transition(U_stacked)
        assert sys.rho == expected_vals[3]
        





