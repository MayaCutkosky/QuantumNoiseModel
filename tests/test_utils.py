'''
Tests originally made by me (Maya Cutkosky), but supplemented a lot by (curated and sometimes modified) chatgpt outputs.
'''

import pytest
import numpy as np
import jax.numpy as jnp
from utils import Operator, density_matrix, kraus, System, operator, DensityMatrix

from utils import pauli



# --- Fixtures ---
@pytest.fixture
def U_numpy():
    """Pauli-Y operator with NumPy backend."""
    return operator([[0,1],[-1,0]], data_object="numpy")

@pytest.fixture
def U_jax():
    """Pauli-Y operator with JAX backend."""
    return operator([[0,1],[-1,0]], data_object="jax")

@pytest.fixture
def K_numpy():
    """Simple Kraus operator (2 Kraus elements for a qubit)."""
    # Identity and Pauli-X as Kraus elements
    elements = np.stack([np.eye(2), np.array([[0,1],[1,0]])])
    return kraus(elements, data_object="numpy")

@pytest.fixture
def K_jax():
    """Simple Kraus operator with JAX backend."""
    elements = np.stack([np.eye(2), np.array([[0,1],[1,0]])])
    return kraus(elements, data_object="jax")


@pytest.mark.dependency(name = 'operator')
class OperatorTests:
    def test_operator_data_type(self, U):
        assert U.find_data_type() in ("numpy", "jax")

    def test_multiplication(self, U):
        U2 = U * U
        assert U2.shape == (2,2)
        # Pauli-Y squared = -I
        expected = -1 * np.eye(2)
        assert np.allclose(np.array(U2._data), expected)

    def test_add_sub(self, U):
        assert np.allclose(np.array((U - U)._data), 0)
        assert np.allclose(np.array((U + U)._data), 2*np.array(U._data))

    def test_scalar_ops(self, U):
        O = 2 * U
        assert np.allclose(np.array(O._data), 2*np.array(U._data))
        O2 = U / 2
        assert np.allclose(np.array(O2._data), np.array(U._data) / 2)

    def test_noncommutative(self, U):
        X = operator([[0,1],[1,0]], data_object=U.data_object)
        Z = operator([[1,0],[0,-1]], data_object=U.data_object)
        assert not np.allclose(np.array((X*Z)._data), np.array((Z*X)._data))

    def test_tensor_shapes(self, U):
        I = operator(np.eye(2), data_object=U.data_object)
        O1 = U.tensor(I)
        O2 = U.rtensor(I)
        assert O1.shape == (4,4)
        assert O2.shape == (4,4)
        assert not np.allclose(np.array(O1._data), np.array(O2._data))

    def test_tensor_with_identity(self, U):
        I = operator(np.eye(2), data_object=U.data_object)
        O = U.tensor(I)
        assert np.allclose(np.array(O._data), np.kron(np.array(U._data), np.eye(2)))

    def test_large_operator(self, U):
        mat = np.random.randn(4,4) + 1j*np.random.randn(4,4)
        O = operator(mat, data_object=U.data_object)
        assert O.shape == (4,4)
        assert np.allclose(np.array(O._data), np.array(mat))

    def test_invalid_division(self, U):
        with pytest.raises(Exception):
            _ = U / U
 
@pytest.mark.dependency(name = 'Kraus')
class Test_Kraus:
    
    def test_kraus_multiplication(self):
        np.random.seed(0)
        U = kraus(np.random.rand(3,2,2))
        rho = density_matrix(np.random.rand(2))
        assert U * rho == [u * rho for u in U]
    
    def test_is_Kraus_function_true(self):
        K = kraus([pauli['X']*0.5, pauli['Y']*0.5, pauli['Z'] * 0.5, pauli['I'] * 0.5])
        assert K.is_Kraus()
    
    def test_is_Kraus_function_false(self):
        K = kraus([pauli['X']*0.5, pauli['Y']*0.5, pauli['Z'] * 0.5, pauli['I'] * 0.5])
        assert K.is_Kraus()




@pytest.fixture
def rho_single():
    """Single-qubit density matrix |0><0|."""
    ket0 = np.array([1, 0])
    return density_matrix(ket0)


@pytest.fixture
def rho_bell():
    """Two-qubit Bell state density matrix (|00> + |11>)/√2."""
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    return density_matrix(psi)

def random_density_matrix(seed, dim, data_object = None):
    """Generate a random valid density matrix."""
    np.random.seed(seed)
    return density_matrix(np.random.rand(dim), data_object = data_object)

def make_random_density_matrix_params(num = 1, max_dims = 8):
    lst = []
    seed = 0
    dim_list = [2,4,8]
    if max_dims == 2:
        dim_list = [2]
    for data_object in [None, 'jax']:
        for dim in dim_list:
            for i in range(3):
                if num == 1:
                    lst.append( random_density_matrix(seed, dim, data_object) )
                else:
                    sub_lst = []
                    for j in range(num):
                        sub_lst.append( random_density_matrix(seed, dim, data_object) )
                    seed += 1
                    lst.append( tuple(sub_lst))
    return lst

import itertools as it
@pytest.mark.dependency(name = 'DensityMatrix')
class Test_DensityMatrix:
    @pytest.mark.parametrize('a,b,c',make_random_density_matrix_params(3, max_dims = 2))    
    def test_partial_trace(self, a,b,c):
        rho = a.tensor(b).tensor(c)
        assert b.tensor(c) == rho.partial_trace(0)
        assert a.tensor(c) == rho.partial_trace(1)
        assert a.tensor(b) == rho.partial_trace(2)
    

    def test_tensor_and_partial_trace_single(self, rho_single):
        rho = rho_single.tensor(rho_single)
        assert isinstance(rho, DensityMatrix)
        # Partial trace over first qubit should yield |0><0|
        reduced = rho.partial_trace(0)
        assert np.allclose(np.array(reduced._data), np.array(rho_single._data))

    def test_partial_trace_multiple(self, rho_bell):
        # Trace out qubit 0: should leave maximally mixed state on qubit 1
        reduced = rho_bell.partial_trace([0])
        expected = 0.5 * np.eye(2)
        assert np.allclose(np.array(reduced._data), expected)

    def test_transition_unitary(self, rho_single):
        X = operator([[0,1],[1,0]])  # Pauli-X
        rho_out = rho_single.transition(X)
        # Should map |0><0| → |1><1|
        expected = np.array([[0,0],[0,1]])
        assert np.allclose(np.array(rho_out._data), expected)

    def test_transition_kraus(self, rho_single):
        # Bit-flip channel: {√p X, √(1-p) I}
        p = 0.3
        K0 = np.sqrt(1-p) * np.eye(2)
        K1 = np.sqrt(p) * np.array([[0,1],[1,0]])
        K = kraus(np.stack([K0,K1]))
        rho_out = rho_single.transition(K)
        # Should yield mixed state: (1-p)|0><0| + p|1><1|
        expected = np.array([[1-p, 0],[0, p]])
        assert np.allclose(np.array(rho_out._data), expected)

    def test_transition_qubit(self, rho_bell):
        # Apply Z gate on qubit 0
        Z = operator([[1,0],[0,-1]])
        rho_out = rho_bell.transition_qubit(Z, [0])
        # Expect same spectrum, just phase flip
        assert np.allclose(
            np.linalg.eigvals(np.array(rho_bell._data)),
            np.linalg.eigvals(np.array(rho_out._data))
        )

    def test_invalid_num_qubits(self):
        mat = np.eye(3)  # not power of 2
        with pytest.raises(AssertionError):
            _ = DensityMatrix(buffer=mat)
            
        
    @pytest.mark.parametrize('rho',make_random_density_matrix_params(1))    
    def test_init_with_valid_matrix(self,rho):
        assert np.allclose(rho.trace(), 1.0)
        assert np.all(np.linalg.eigvals(rho._data) >= -1e-7)  # PSD
    
    
    @pytest.mark.parametrize('dim1, dim2', it.product([2,4,8], [2,4,8] )  )  
    def test_tensor_product_density_matrix(self, dim1, dim2):
        dm1 = random_density_matrix(0, dim1)
        dm2 = random_density_matrix(0, dim2)
        dm12 = dm1.tensor(dm2)
    
        assert dm12.shape == (dim1 * dim2,  dim1 * dim2)
        assert np.allclose(dm12.trace(), 1.0)
    
    @pytest.mark.parametrize('rho',make_random_density_matrix_params(1)) 
    def test_adjoint_is_hermitian(self,rho):
        assert rho ==  rho.adjoint()
    
    @pytest.mark.parametrize('rho',make_random_density_matrix_params(1)) 
    def test_unitary_evolution_preserves_density_matrix(self,rho):
        
        dim= rho.shape[0]
        # Random unitary via QR
        X = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim )
        Q, _ = np.linalg.qr(X)
        U = Q
        assert np.allclose( np.matmul( U , np.transpose(U.conjugate() ) ) , np.identity(dim) )
    
        evolved_rho = rho.transition(operator(U, data_object = rho.data_object))
    
        assert np.allclose(evolved_rho.trace(), 1.0)
        assert np.all(np.linalg.eigvals(evolved_rho._data) >= -1e-7)
        assert evolved_rho == evolved_rho.adjoint()
        assert np.allclose( (evolved_rho * evolved_rho ).trace(), 1.0)


from utils import ideal_gates


def h_inst(sys, i):
    sys = sys.transition_qubit(ideal_gates['sx'], (i, ) )
    sys = sys.transition_qubit(ideal_gates['rz'](np.pi/2), (i, ) )
    sys = sys.transition_qubit(ideal_gates['sx'], (i, ) )
    return sys
    
def x_inst(sys, i):
    return sys.transition_qubit(ideal_gates['x'], (i, ) )
    
def cnot_inst(sys, i,j):
    sys = h_inst(sys,1)
    sys = sys.transition_qubit(ideal_gates['cz'],(0,1))
    sys = h_inst(sys,1)
    return sys


#@pytest.mark.dependency(name = 'system',  depends = ['operator', 'DensityMatrix'])
# @pytest.mark.parametrize('U1, U_stacked, expected_vals',[
#     (
#          operator([[1,1],[1,-1]], data_object = 'jax') / np.sqrt(2),
#          operator([[[1,1],[1,-1]]]*3,  data_object = 'jax') / np.sqrt(2),
#          [
#              density_matrix([[[1/2,1/2],[1/2,1/2]]]*3),
#              [density_matrix([[[1,0],[0,0]],[[1/2,1/2],[1/2,1/2]],[[1,0],[0,0]]]), density_matrix([[[1,0],[0,0]]] * 3)],
#              density_matrix([[[1/2,1/2],[1/2,1/2]],[[1,0],[0,0]],[[1/2,1/2],[1/2,1/2]]]),
#              density_matrix([[[1/2,1/2], [1/2,1/2]]]*3)
#          ]
#     ),
#     (
#          kraus([[[1,1],[1,-1]], [[0,1],[1,0]], [[1,0],[0,1]]],  data_object = 'jax') / 2,
#          kraus([[[[1,1],[1,-1]], [[0,1],[1,0]], [[1,0],[0,1]]]] * 3,  data_object = 'jax') / 2,
#          [
#              density_matrix([[[.5,.25], [.25,.5]]]*3),
#              [density_matrix([[[1,0],[0,0]],[[.5,.25], [.25,.5]],[[1,0],[0,0]]]), density_matrix([[[1,0],[0,0]], [[0.625,0.125],[0.125,0.375]],[[1,0],[0,0]]])],
#              density_matrix([[[.5,.25], [.25,.5]],[[1,0],[0,0]],[[.5,.25], [.25,.5]]]),
#              density_matrix([[[.5,.25], [.25,.5]]]*3)
#          ]
#     )
# ])    
class Test_system:
    @staticmethod
    def convert_expected_val(val):
        output = val[0]
        for i in range(3):
            output = output.tensor(val[i+1])
        return output

    @pytest.mark.parametrize('U1, expected_vals',[
        (
             operator([[1,1],[1,-1]], data_object = 'jax') / np.sqrt(2),
             density_matrix([[[1/2,1/2],[1/2,1/2]]]*3, data_object = 'jax')
        ),
        (
             kraus([[[1,1],[1,-1]], [[0,1],[1,0]], [[1,0],[0,1]]],  data_object = 'jax') / 2,
             density_matrix([[[.5,.25], [.25,.5]]]*3, data_object = 'jax')
        )
    ])    
    def test_system_transition_single_qubit(self,U1, expected_vals):
        sys = System(3)
        sys = sys.transition_qubit(U1,(1,))
        expected_vals0 = expected_vals[0]
        expected_vals1 = expected_vals[0]
        for i in range(2):
            expected_vals0= expected_vals0.tensor(expected_vals[1][0][i+1])
            expected_vals1= expected_vals1.tensor(expected_vals[1][1][i+1])
        assert sys.rho == expected_vals0
        sys = sys.transition_qubit(U1, (1,))
        assert sys.rho == expected_vals1
    
    
    def test_system_transition(self,U1,U_stacked, expected_vals):
        sys = System(3)
        sys = sys.transition(U_stacked)
        assert sys.rho == self.convert_expected_val(expected_vals[3])
        




