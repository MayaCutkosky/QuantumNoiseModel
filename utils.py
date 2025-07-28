import numpy as np
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

def operator(x, data_object = None):
    n = len(x)
    M = Operator([n,n], buffer= np.array(x, dtype = complex), data_object=data_object)
    return M
def kraus(x, **kwargs):
    M = Kraus(buffer= x, **kwargs)
    return M

def digit_ind_to_binary_inds(n):
    return np.tile( np.arange(2**n), [1,1]).T // 2 ** (n - 1 - np.tile(np.arange(n-1,-1,-1), [1,1]) ) % 2

def binary_inds_to_digit_ind(inds):
    n = len(inds[0])
    return np.sum(2 ** np.arange(n-1, -1, -1) * inds,axis=1)

class Operator:
    def __init__(self, shape = [2,2], dtype = complex, buffer = None, data_object = None):
        self.type = "operator"
        self._data = None
        if buffer is not None:
            if isinstance(buffer, Operator):
                buffer = buffer._data
            elif isinstance(buffer, list):
                data = []
                for x in buffer:
                    if hasattr(x, 'tolist'):
                        data.append(x.tolist())
                    else:
                        data.append(x)
                    if data_object is None:
                        data_object = self.find_data_type(x)
                    elif data_object == "numpy":
                        if self.find_data_type(x) == 'jax':
                            data_object = 'jax'
                buffer = data
        if data_object is None:
            if buffer is None:
                self._data_object = "numpy"
            else:
                self.data_object = self.find_data_type(buffer)
                if self.data_object is None:
                    self.data_object = "numpy"
        else:
            self.data_object = data_object
        if self.data_object == 'jax':
            if buffer is None:
                self._data = jnp.zeros(shape, dtype = dtype)
            else:
                self._data = jnp.array(buffer, dtype = dtype)
        elif self.data_object == 'numpy':
            self._data = np.array(buffer, dtype = dtype)
        else:
            Exception("Not understood data type:", data_object)
        
        for attr in ['shape', 'real', 'imag']:
            setattr(self, attr, getattr(self._data, attr))
        if self.data_object == 'jax':
            self.np = jnp
        else:
            self.np = np

    def find_data_type(self, data = None):
        if data is None:
            data = self._data
        data_object = str(type(data))
        if "numpy" in data_object:
            return "numpy"
        elif "jax" in data_object:
            return "jax"
        return None

    def _create_new(self, **kwargs):
        output = self.__new__(type(self))
        output.__init__(**kwargs)
        return output

    def __mul__(self,y):
        data_object = self.data_object
        if isinstance(y, Operator):
            if y.data_object == 'jax':
                data = jnp.matmul(self._data, y._data)
                data_object = 'jax'
            else:
                data = self.np.matmul(self._data,y._data)
        else:
            data = self._data * y
        return self._create_new(buffer = data)

    def __rmul__(self, y):
        if isinstance(y, Operator): #will never be true!
            data = self.np.matmul(y._data, self._data)
        else:
            data = y * self._data
        return self._create_new(buffer = data)

    def _replace_binary_method(self,fun_name,y):
        fun = getattr(self._data, fun_name)
        if isinstance(y, Operator):
            data = fun(y._data)
        else:
            data = fun(y)
        return self._create_new(buffer = data)

    def __sub__(self,y):
        return self._replace_binary_method("__sub__", y)


    def __add__(self, y):
        return self._replace_binary_method('__add__', y)

    def __truediv__(self, y):
        #Better not to use division for operator and operator. Only sensible interpretation is multiply by inverse which is less confusing to just code. Could use for constants though. And this is used later.
        if isinstance(y, Operator):
            return Exception('Operator-Operator division not allowed. Use Operator * Operator.adjoint()' )
        else:
            return self._create_new(buffer = self._data / y )

    def __floordiv__(self, y): #don't actually see myself using this...
        if isinstance(y, Operator):
            return Exception('Operator-Operator division not allowed. Use Operator * Operator.adjoint()' )
        else:
            return self._create_new(buffer = self._data // y )

    def tensor(self, y):
        if isinstance(y, Operator):
            y_data = y._data
        else:
            y_data = y
        data = self.np.kron(self._data,y_data)
        if isinstance(y, Kraus):
            return y._create_new(buffer = data)

        return self._create_new(buffer = data)
    def rtensor(self, y):
        if isinstance(y, Operator):
            y_data = y._data
        else:
            y_data = y
        data = self.np.kron(y_data, self._data)
        if isinstance(y, Kraus):
            return y._create_new(buffer = data)

        return self._create_new(buffer = data)
        

    def adjoint(self):
        data = self.np.swapaxes(self.np.conj(self._data),-1,-2)
        return self._create_new(buffer = data)

    def is_unitary(self):
        return self * self.adjoint() == np.identity(len(self))

    def __eq__(self,x):
        if isinstance(x, list):
            for xi, yi in zip(x, self):
                if not yi == xi:
                    return False
            return True
        if isinstance(x, Operator):
            x = x._data
        return self.np.max(np.abs(self._data - x)) < 1e-10

    def __len__(self):
        return len(self._data)

    def __getitem__(self,i):
        return self._create_new(buffer = self._data[i])
    def __setitem__(self, i, value):
        if isinstance(value, Operator):
            value = value._data
        
        if self.data_object == 'jax':
            self._data = self._data.at[i].set(value)
        elif self.data_object == 'numpy' :
            self._data[i] = np.array(value)
    def __iter__(self):
        for x in self._data:
            yield self._create_new(buffer = x)

    def _replace_numpy_method(self,fun_name,*args, **kwargs):
        fun = getattr(self.np,fun_name)
        data = fun(self._data, *args, **kwargs)
        return self._create_new(buffer = data)

    def sum(self, **kwargs):
        return self._replace_numpy_method('sum', **kwargs)
    def round(self, decimals):
        return self._replace_numpy_method('round',decimals)

    def tolist(self):
        return self._data.tolist()

    def __str__(self):
        return str(self._data)
    def __repr__(self):
        return self.type + '(' + str(self._data) + ')'

    def __call__(self,*args):
        return self._data[*args]
    
    def swapaxes(self, axis1, axis2):
        self._data = self.np.swapaxes(self._data, axis1, axis2)
        self.shape = self._data.shape
        

        

class Kraus(Operator):
    def __init__(self, shape = [1,2,2], *args, **kwargs):
        super().__init__(shape, *args, **kwargs)
        self.type = 'kraus'
    def tensor(self, y):
        if isinstance(y, Operator):
            y = y._data
        if isinstance(y, Kraus):
            data = self.np.vectorize(self.np.kron)(self._data, y)
        else:
            data = self.np.kron(self._data,y)
        return self._create_new(buffer = data)
    def extend(self, y):
        if isinstance(y, Operator):
            if y.data_object == 'jax':
                self.np = jnp
            if isinstance(y, Kraus):
                y = self.np.split(y._data, len(y))
        x = self.np.split(self._data, len(self))
        self._data = self.np.vstack(x + y)
    def is_Kraus(self):
        return np.sum(self * self.adjoint(),axis=-3) == np.identity(self.shape[-1])


class DensityMatrix(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.type = 'density_matrix'
        self.num_qubits = np.log2(self.shape[-1])
        assert self.num_qubits % 1 == 0
        self.num_qubits = int(self.num_qubits)
    def transition(self,U : Operator):
        is_Kraus = isinstance(U,Kraus)
        if is_Kraus and len(U.shape) > 3:
            self._data = self.np.expand_dims(self._data, -3)
        output = self._create_new(buffer = (U * self * U.adjoint())._data)
        if is_Kraus:
            if len(U.shape) > 3:
                self._data = self.np.reshape(self._data, self.shape)
            output = output.sum(axis = -3)
        # trace = jnp.trace(output._data, 0,-1,-2)
        # if trace.shape == ():
        #     trace = jnp.tile(trace, [self.shape[-1], self.shape[-2]])
        # else:
        #     trace = jnp.tile(trace, [self.shape[-1], self.shape[-2],1]).T
        # output._data = output._data / trace
        return output
    
    def transition_qubit(self,U, qubits):
        qubits = list(qubits)
        qubits.sort()
        if len(qubits) == 2:
            U0, U1 = U
            mid_qubits = np.identity(int(2 ** int(qubits[1] - qubits[0] - 1)))
            op = operator([[1,0],[0,0]]).tensor(mid_qubits).tensor(U0) 
            op += operator([[0,0],[0,1]]).tensor(mid_qubits).tensor(U1)
        else:
            op = U
        op = operator(np.identity(2 ** qubits[0])).tensor(op).tensor(operator(np.identity(2 ** (self.num_qubits - qubits[-1]-1))))
        return self.transition(op)


    def partial_trace(self, i):
        if isinstance(i, list):
            l = i
            l .sort()
            l.reverse()
            rho = self._create_new(buffer = self._data)
            for i in l:
                rho = rho.partial_trace(i)
            return rho
        #shape : s, i,j
        n = int(2 ** i)
        m = len(self) // (2*n)

        ind = np.tile(np.arange(m),n) + np.arange(n).repeat(m)*2*m
        i1 = np.tile(ind,[len(ind),1])
        j1 = i1.T
        i2 = i1 + m
        j2 = i2.T

        return self[...,j1,i1] + self[...,j2,i2]

    def measure(self, psi):
        output = self.np.matmul(self.np.matmul([psi], self),psi)
        assert output.imag == 0
        return output.real[0]

#hacky...
@register_pytree_node_class
class JaxDensityMatrix(DensityMatrix):
    def __init__(self,buffer):
        self.type = 'density_matrix'
        self._data = buffer
        self.data_object = 'jax'
        self.np = jnp
        self.set_attr()
    
    def set_attr(self):
        
        for attr in ['real', 'imag']:
            setattr(self, attr, getattr(self._data, attr,0.))
        setattr(self, 'shape', getattr(self._data, 'shape',(1,1)))
        self.num_qubits = np.log2(self.shape[-1])
        self.num_qubits = int(self.num_qubits)
    
    def __repr__(self):
        return "DensityMatrix({})".format(self._data)

    def tree_flatten(self):
        
        children = (self._data,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(*children)
        obj.set_attr()
        return obj

def density_matrix(x, **kwargs):
    if len(np.shape(x)) == 1: # x is a state
        #make x a proper state
        x = x / np.linalg.norm(x)
        x = np.matmul(np.expand_dims(x,1),[x])
    if kwargs.get('data_object') == 'jax':
        rho = JaxDensityMatrix(jnp.array(np.array(x, dtype = complex)))
    else:
        rho = DensityMatrix(np.shape(x), complex, np.array(x, dtype = complex), **kwargs)
    assert rho.shape[-1] == rho.shape[-2]
    return rho

from jax import lax
class System:
    def __init__(self,size = None, config = None, **kwargs):
        if config is None:
            self.rho = []
            self.reverse_ind = []
            for i in range(size):
                self.rho.append(density_matrix([[1,0],[0,0]], **kwargs))
                self.reverse_ind.append([i])
            self.inds = np.zeros([size,2], dtype = int)
            self.inds[:,0] = np.arange(size).astype(int)
            self.careful_mode = True
            self.num_qubits = size
        else:
            for key, value in config.items():
                setattr(self, key, value)
    def transition_qubit(self,U, qubits, in_place = True, return_gradient = False):
        '''
        

        Parameters
        ----------
        U : TYPE
            DESCRIPTION.
        qubits : TYPE
            DESCRIPTION.
        in_place : TYPE, optional
            Currently in_place = False is not implimented. The default is True.
        return_gradient : TYPE, optional
            Gradient is dictionary of values with rho as the uncompressed n by n matrix, where n is the number of qubits. The default is False.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if not in_place:
            raise NotImplementedError()
        if len(qubits) == 2:
            ind0, i = self.inds[qubits[0]]
            ind1, j = self.inds[qubits[1]]
            
            if ind0 == ind1:
                self.rho[ind0] = self.rho[ind0].transition_qubit(U, [i,j])
                
            else:
                if ind0 > ind1:
                    rho0, rho1 = self.rho.pop(ind0), self.rho.pop(ind1)
                    reverse_ind0, reverse_ind1 = self.reverse_ind.pop(ind0), self.reverse_ind.pop(ind1)
                else:
                    rho1, rho0 = self.rho.pop(ind1), self.rho.pop(ind0)
                    reverse_ind1, reverse_ind0 = self.reverse_ind.pop(ind1), self.reverse_ind.pop(ind0)
                rho = rho0.tensor(rho1)
                
                j = j + len(reverse_ind0)
                rho = rho.transition_qubit(U, [i,j])
                self.rho.append(rho)
                
                reverse_ind = reverse_ind0 + reverse_ind1
                self.reverse_ind.append(reverse_ind)
                
                for ind, rev_inds in enumerate(self.reverse_ind):
                    for rev_ind in rev_inds:
                        self.inds[rev_ind,0] =  ind
                self.inds[reverse_ind1, 1] = self.inds[reverse_ind1, 1] + len(reverse_ind0)
            
        elif len(qubits) == 1:
            ind, i = self.inds[qubits[0]]
            self.rho[ind] = self.rho[ind].transition_qubit(U, (i,) )
        else:
            raise NotImplementedError()

    def transition(self, U_array):
        '''
            U : list of operators of length self.rho
        '''
        for i, (rho, qubit_inds) in enumerate(zip(self.rho, self.reverse_ind)):
            U = U_array._create_new(buffer = [1])
            for Ui in U_array[qubit_inds]:
                U = U.tensor(Ui)
            self.rho[i] = rho.transition(U)
            
        return self.rho
    def calc_probabilities(self, readout_qubits = None):
        if readout_qubits is None:
            readout_qubits = np.arange(self.num_qubits).tolist()
        prob = 1
        binary_inds = digit_ind_to_binary_inds(len(readout_qubits))
        for rho, qubit_inds in zip(self.rho, self.reverse_ind):
            inds = []
            extra_qubit_inds = []
            for i, q in enumerate(qubit_inds):
                if q in readout_qubits:
                    inds.append(readout_qubits.index(q))
                else: 
                    extra_qubit_inds.append(i)
            if len(extra_qubit_inds):
                rho = rho.partial_trace(extra_qubit_inds)
            local_prob_inds = binary_inds_to_digit_ind( binary_inds[:,inds] )
            local_prob = rho(local_prob_inds, local_prob_inds).real+1e-8
            
            prob *= local_prob
        return prob
        
            
            
            

import itertools as it
pauli = {
    'I' : operator(np.identity(2)),
    'X' : operator([[0,1],[1,0]]),
    'Y' : operator([[0,complex(0,-1)],[complex(0,1),0]]),
    'Z' : operator([[1,0],[0,-1]])
}


def crand(**args):
    return complex(np.random.rand(**args),np.random.rand(**args))
def expi(theta):
    return np.cos(theta) + complex(0,1) * np.sin(theta)


ideal_gates = {
    'id' : operator(np.identity(2)),
    'x' : pauli['X'],
    'sx' : operator([[complex(1,1),complex(1,-1)],[complex(1,-1),complex(1,1)]])/2,
    'rz' : lambda phi : operator([[expi(-phi/2),0],[0,expi(phi/2)]]),
    'cz' : (pauli['I'], pauli['Z']),
}
#cz = np.identity(4)
#cz[3,3] = -1
#ideal_gates['cz'] = operator(cz)

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

def softmax(x,axis = None):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis = axis)

def find_neighboring_qubits(q, coupling_map, distance = 1, return_close_neighbors = False):
    close_neighbors = []
    if isinstance(q, int):
        curr_dist_neighbors = [q]
    else:
        curr_dist_neighbors = q

    for i in range(distance):
        close_neighbors += list(curr_dist_neighbors)
        next_dist_neighbors = set()
        for q_neigh in curr_dist_neighbors:
            for q_far_neigh in coupling_map[q_neigh]:
                if q_far_neigh not in close_neighbors:
                    next_dist_neighbors.add(q_far_neigh)
        curr_dist_neighbors = next_dist_neighbors
    if return_close_neighbors:
        return curr_dist_neighbors, close_neighbors
    else:
        return curr_dist_neighbors


def add_gate(circuit, used_qubits, coupling_map, distance = 1):
    possible_qubits, near_qubits = find_neighboring_qubits(used_qubits, coupling_map, distance, True)
    for q1 in np.random.permutation(list(possible_qubits)):
        q1 = int(q1)
        for q2 in np.random.permutation(list(coupling_map[q1])):
            q2 = int(q2)
            if q2 in near_qubits:
                continue
            if q1 > q2:
                new_gate = ('cz',(q2,q1), None)
            else:
                new_gate = ('cz', (q1,q2), None)
            break
    circuit.insert(np.random.randint(len(circuit)), new_gate)
    used_qubits += [q1,q2]
    return circuit, used_qubits

def add_X_gates(circuit, used_qubits, coupling_map, distance = 1):
    possible_qubits, near_qubits = find_neighboring_qubits(used_qubits, coupling_map, distance, True)
    q = int(np.random.choice(list(possible_qubits)))
    ind = np.random.randint(len(circuit))
    new_gate = ('x',(q,), None)
    circuit.insert(ind, new_gate)
    circuit.insert(ind, new_gate)
    used_qubits.append(q)
    return circuit, used_qubits
