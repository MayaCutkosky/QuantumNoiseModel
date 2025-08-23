import jax.numpy as jnp
import optax
from jax import value_and_grad, clear_caches, random, custom_gradient, jit
import json
from data import process_backend
import numpy as np
from utils import System, ideal_gates, kraus, softmax, pauli, operator, density_matrix, JaxDensityMatrix
from gate_utils import add_gate
import pickle
import itertools as it
from equinox import Module
import jax
from jax_utils import map_qubits
class RandomGenerator:
    def __init__(self):
        self.key = random.PRNGKey(10)
        self.i = 0
    def get_subkey(self):
        self.i += 1
        return random.fold_in(self.key, self.i)
    def random_int(self, p, n):
        self.get_subkey()
        return np.random.default_rng(self.i).choice(np.arange(n), p = p)
        subkey = self.get_subkey()
        return random.choice(subkey, jnp.arange(n),p=p).tolist()

    def random_choice(self, choices):
        subkey = self.get_subkey()
        i = random.randint(subkey, (), len(choices)-1, 0)
        return choices[i]
    
    def boolean(self, p):
        r = random.uniform(self.get_subkey())
        r = np.random.default_rng(self.i).uniform()
        return r > p
    def reset(self):
        self.__init__()



def tensor_prod_deriv(dC, A, B):
    '''
    Finds derivatives for C = A tensor B

    Parameters
    ----------
    dC : jnp.array of shape  (m*n, m*n)
    A : jnp.array of shape (m,m)
    B : jnp.array of shape (n,n)

    Returns
    -------
    dA : jnp.array of shape [m,m]
    dB : jnp.array of shape (n,n)

    '''
    m = A.shape[0]
    n = B.shape[0]
    
    i,j,u,v = jnp.indices([m,m,n,n])
                 
    dA = jnp.sum( B._data * dC(u+n*i, v+n*j),(2,3) ).T
    
    dB = jnp.sum( A._data * dC(i.T * m + u.T, j.T *m + v.T), (2,3) ).T
    return A._create_new(buffer = dA), B._create_new(buffer = dB)

def partial_trace_derviative(dA, dB):
    '''
    Finds derivate for A, B = Tr_B(C), Tr_A(C)
    '''
    m = dA.shape[0]
    n = dB.shape[0]
    
    return dA.tensor(jnp.identity(n)) +dB.rtensor(jnp.identity(m))
#ToDo: Use jax.tree_map to vectorize
#\rho_i = sum x_j g_j( h_i (\rho_{i-1} ) )
def calc_del_g_del_h(i,j,n):
    size = [2 **i, 2 ** (j-i-1), 2 ** (n - j-1)]
    i,j,k = np.indices(size)
    diag_neg_vals = (size[1] * size[2]* 2  + size[2] + k + j * size[2]* 2 + i * size[1] * size[2] * 4).ravel()
    diag_elements = np.ones(2**n)
    diag_elements[diag_neg_vals] = -1
    return jnp.array( np.outer(diag_elements, diag_elements) )



def calc_d_fi(df_i, rho, connections, reverse_ind, k0, x):
    
    #will have to fix this, but first
    dx = [ ]
    dh = [rho * x[k0]  for rho in df_i]
    for k, ( (ind0, i), (ind1,j) ) in enumerate(connections):
        if k == k0:
            dx.append( sum([(df_i_l._data * rho_l._data).sum() for df_i_l, rho_l in zip(df_i, rho)] ) )
            continue
        dx_k = sum([(df_i_l._data * rho_l._data).sum() for l, (df_i_l, rho_l) in enumerate(zip(df_i, rho)) if l != ind0 and l != ind1] )
        if i > j:
            i,j = j,i
            ind0,ind1 = ind1,ind0
        if ind0 == ind1:
            dx_k += (rho[ind0].transition_qubit(ideal_gates['cz'],(i,j))._data * df_i[ind0]._data).sum()
            del_g_del_h = calc_del_g_del_h(i,j,dh[ind0].num_qubits )          
            dh_ind0 = dh[ind0] * x[k] * del_g_del_h
            
            dh = [rho0 * x[k] + rho1 for l, (rho0, rho1) in enumerate(zip(df_i, dh)) ]
            dh[ind0] = dh_ind0

        else:
            j = j + reverse_ind[ind0]
            dx_k += (rho[ind0].tensor(rho[ind1]).transition_qubit(ideal_gates['cz'],(i,j))._data * df_i[ind0].tensor(df_i[ind1])._data).sum()
                            
            d_rho0_tensor_rho1 = partial_trace_derviative(df_i[ind0], df_i[ind1]) 
            
            del_g_del_h = calc_del_g_del_h(i,j, d_rho0_tensor_rho1.num_qubits)     
            d_rho0_tensor_rho1 = d_rho0_tensor_rho1 * x[k] * del_g_del_h
            dh_ind0, dh_ind1 = tensor_prod_deriv(d_rho0_tensor_rho1, rho[ind0], rho[ind1])
            
            dh = [rho0 * x[k] + rho1 for l, (rho0, rho1) in enumerate(zip(df_i, dh)) ]
            dh[ind0] = dh_ind0 + dh[ind0]
            dh[ind1] = dh_ind1 + dh[ind1]
        dx.append(dx_k)
    return jnp.stack(dx).real, dh



#f_i should be differentiated under the assumption that all possible crosstalk happened, not just the chosen one. But only apply the chosen one.
def f_i(sys,  x, k,k0, connections, x_prime):
    
    @custom_gradient
    def f_i_diff_fun(x, rho, x_prime):
        def grad_fn(df_i):
            dx,dh = calc_d_fi(df_i, rho, connections, reverse_ind, k0, x_prime)
            return None, dh, dx

        x_k = x[k] #can fix if objects.
        rho_new = [r for r in rho]
        rho_new[ind] =  (1-x_k) * rho[ind] + x_k * rho[ind].transition_qubit(ideal_gates['cz'], [i,j]) 
        return rho_new, grad_fn

    @custom_gradient
    def f_i_diff_fun2(rho, x_prime):
        def grad_fn(df_i):
            dx,dh = calc_d_fi(df_i, rho, connections, reverse_ind, k0, x_prime)
            return dh, dx
        return rho, grad_fn

    
    
    if k != k0:
        qubits = connections[k]
        ind0, i = sys.inds[qubits[0]]
        ind1, j = sys.inds[qubits[1]]
        if ind0 == ind1:
            ind = ind0
            rho = sys.rho[ind] 
        
        else:
            if ind0 > ind1:
                rho0, rho1 = sys.rho.pop(ind0), sys.rho.pop(ind1)
                reverse_ind0, reverse_ind1 = sys.reverse_ind.pop(ind0), sys.reverse_ind.pop(ind1)
            else:
                rho1, rho0 = sys.rho.pop(ind1), sys.rho.pop(ind0)
                reverse_ind1, reverse_ind0 = sys.reverse_ind.pop(ind1), sys.reverse_ind.pop(ind0) 
            j = j + len(reverse_ind0)
            
            rho = rho0.tensor(rho1)
            ind = len(rho)
            sys.rho.append(rho)
                
            reverse_ind = reverse_ind0 + reverse_ind1
            sys.reverse_ind.append(reverse_ind)
                
            for ind, rev_inds in enumerate(sys.reverse_ind):
                for rev_ind in rev_inds:
                    sys.inds[rev_ind,0] =  ind
            sys.inds[reverse_ind1, 1] = sys.inds[reverse_ind1, 1] + len(reverse_ind0)
        connections = sys.inds[connections].tolist()
        reverse_ind = [len(i) for i in sys.reverse_ind]
        sys.rho = f_i_diff_fun(x, sys.rho, x_prime)
    else:
        connections = sys.inds[connections].tolist()
        reverse_ind = [len(i) for i in sys.reverse_ind]
        sys.rho =  f_i_diff_fun2(sys.rho, x_prime)

def compute_prob_matrix(M): #from cursor
    n = M.shape[0]
    size = 2 ** n

    # Generate all possible n-bit vectors for A and B
    A_bits = ((jnp.arange(size)[:, None] >> jnp.arange(n-1, -1, -1)) & 1).astype(np.uint8)  # (size, n)
    B_bits = ((jnp.arange(size)[:, None] >> jnp.arange(n-1, -1, -1)) & 1).astype(np.uint8)  # (size, n)

    # Expand A_bits and B_bits to shape (size, size, n)
    A_expand = A_bits[:, None, :]  # (size, 1, n)
    B_expand = B_bits[None, :, :]  # (1, size, n)

    # Compute probabilities for each digit position
    mask_A1 = (A_expand == 1)
#    mask_A0 = ~mask_A1

    prob_A1 = jnp.where(B_expand == 0, M[:, 0], 1 - M[:, 0])  # (1, size, n)
    prob_A0 = jnp.where(B_expand == 1, M[:, 1], 1 - M[:, 1])  # (1, size, n)

    probs = jnp.where(mask_A1, prob_A1, prob_A0)  # (size, size, n)

    M_prime = jnp.prod(probs, axis=2)  # (size, size)

    return M_prime

random_gen = RandomGenerator()
class Model(Module):
    num_qubits : int
    num_gates : int
    connections : list
    error_operators : dict
    readout_err : np.array
    cross_talk_probabilities : jax.Array
    represented_terms : list
    coupling_map : dict
    def __init__(self, backend = None, backend_properties = None, config = None):
        if config is not None:
            self.load_config(config)
        else:
            if backend_properties is None:
                backend_properties = backend.properties()
            (
                self.num_qubits,
                self.num_gates,
                self.connections,
                self.error_operators,
                self.readout_err,
            ) = process_backend(backend_properties)
            self.initialize_params()
                 
        self.represented_terms = ['num_qubits', 'connections', 'cross_talk_probabilities']
        
        self.coupling_map = dict()
        for q1, q2 in self.connections:
            self.coupling_map.setdefault(q1, set() )
            self.coupling_map.setdefault(q2, set() )
            self.coupling_map[q2].add(q1)
            self.coupling_map[q1].add(q2)
        for key in self.coupling_map:
            self.coupling_map[key] = tuple(self.coupling_map[key])

    def initialize_params(self, noise1 = 0.01, noise2 = 10):
        r = np.random.rand( len(self.connections), len(self.connections)) / self.num_qubits * noise1
        r[np.arange(len(self.connections)), np.arange(len(self.connections))] = noise2        
        self.cross_talk_probabilities = jnp.array(r)
    
    def transition_depol_err(self,sys, gate_type, qubit_ids, sys_qubit_ids):
        depol_err = self.error_operators[gate_type][qubit_ids]['depol']
        
        if random_gen.boolean(depol_err):
            return #no error
        depol_gates = random_gen.random_choice(list(it.combinations(pauli, len(sys_qubit_ids)))[1:])
        for gate, q in zip(depol_gates, sys_qubit_ids):
            sys.transition_qubit(pauli[gate], (q,))

        
    def _run_instruction(self, sys, instruction, used_qubits, cross_talk_prob_params):
        sys_qubit_ids = instruction.get_sys_qubit_ids(used_qubits)
        ideal_operator = instruction.get_operator()
        sys.transition( self.error_operators[instruction.gate_type()][instruction.qubit_ids]['relax'][ np.array(used_qubits) ] )
        sys.transition_qubit(ideal_operator, sys_qubit_ids)
        
        self.transition_depol_err(sys, instruction.gate_type(), instruction.qubit_ids, sys_qubit_ids)
        if len(instruction.qubit_ids) < 2:
            return sys
        #$\\rho^{(1)} = (1 - \\sum_{g' \\in G} p_{g,g'}) C_g R_T \\rho^{(0)} R_T^\\dagger C_g^{\\dagger} + \\sum_{g' \\in G}  p_{g,g'} C_{g'}C_g R_T \\rho^{(0)} R_T^\\dagger (C_gC_{g'})^{\\dagger}  $
        
        ind = np.isin(self.connections,used_qubits).prod(1)
        connections =  [qubits for (i, qubits) in zip(ind, self.connections) if i]
        j = connections.index(instruction.qubit_ids)
        probs = cross_talk_prob_params[j].at[j].set(0)
        
#        i = self.random_gen.random_int([1 - self.prob_perform_cross_talk* (len(connections)-1) ] +[self.prob_perform_cross_talk] * (len(connections)-1), len(connections) )
        #x_prime = self.prob_perform_cross_talk * probs
        
        rho = JaxDensityMatrix(sys.rho[0])
        for cross_talk_qubits, p in zip( connections, probs):
            sys_cross_talk_qubits = [used_qubits.index(q) for q in cross_talk_qubits]                
            rho_i = sys.rho[0].transition_qubit(ideal_gates['cz'], sys_cross_talk_qubits)
            rho_i = rho_i - sys.rho[0]
            
            rho += rho_i * p 
        sys.rho[0] = rho
        return sys
            
            

    def run(self, circuit, readout_qubits, used_qubits = None) -> np.ndarray:
        return self._run(circuit, readout_qubits, used_qubits, self.prepare_params(self.cross_talk_probabilities, used_qubits))
    def _run(self, circuit, readout_qubits, used_qubits, params):
        output= 0
        readout_err_mat = compute_prob_matrix(self.readout_err[readout_qubits])

        readout_qubits =  map_qubits(used_qubits, readout_qubits)
        used_qubits = list(used_qubits)
        random_gen.reset()
        for i in range(1):
            sys = System(len(used_qubits), data_object = 'jax')
            for i in range(len(used_qubits)-1):
                sys.transition_qubit([pauli['I'], pauli['I']], (i, i+1))
            for j, instruction in enumerate(circuit):
                sys = self._run_instruction(sys, instruction, used_qubits, params)
            readout_probs = sys.calc_probabilities(readout_qubits)
            readout_probs = jnp.matmul(readout_probs, readout_err_mat)
            output += readout_probs
        return output
        

    @staticmethod
    def normalize_params(params):
        exp_x = jnp.exp(params).T
        return (exp_x / exp_x.sum(0)).T
    
    def prepare_params(self, params, used_qubits):
        ind = np.where(np.isin(self.connections,used_qubits).prod(1))[0]
        probs = self.normalize_params(params)[ind][:,ind]
        params = probs
        
   #     self.prob_perform_cross_talk = 1 / 2 / len(ind)
        
#        x0 = jnp.diag(probs)
  #      params = probs / self.prob_perform_cross_talk
#        params[np.arange(len(ind)), np.arange(len(ind))] = 1 - self.prob_perform_cross_talk - x0
        return params
    
    
    def calculate_log_likelihood(self,sample):
        (circuit, readout_qubits, used_qubits), exp_readout = sample
        log_pred_readout = jnp.log(self._run(circuit, readout_qubits, used_qubits, self.normalize_params(self.cross_talk_probabilities))+1e-8)
        return np.sum(np.array(exp_readout) * log_pred_readout)
 
        
    def _calculate_loss(self, params, sample_inp, exp_readout):
        circuit, readout_qubits, used_qubits = sample_inp
        
        params = self.prepare_params(params, used_qubits)
        pred_readout_probs = self._run(circuit, readout_qubits, list(used_qubits),  params)
        
        false_circuit, used_qubits = add_gate(list(circuit), list(used_qubits), self.coupling_map)
        
        false_pred_readout_probs = self._run(false_circuit, readout_qubits, list(used_qubits),  params)
        
        exp_readout = exp_readout/exp_readout.sum()
        
        loss = jnp.sum(exp_readout * jnp.log( exp_readout / pred_readout_probs + 1e-8) )
        loss += 10 * jax.nn.relu(jnp.sum( jnp.log( false_pred_readout_probs / pred_readout_probs +1e-8) ) )
        # return readout_probs.sum()
#        log_pred_readout = jnp.log(readout_probs + 1e-9) #deal with prob = 0    
#        log_exp_readout = jnp.log((exp_readout/exp_readout.sum())+1e-9)
#        loss =  jnp.sum( jnp.square(log_exp_readout - log_pred_readout) )
        # cross_talk_probs = params.at[np.arange(len(params)), np.arange(len(params))].multiply(0)#.subtract(params[np.arange(len(params)), np.arange(len(params))])
        
        return loss #+ self.regularization_fun(cross_talk_probs)

    def calculate_loss(self, sample):
        '''
        

        Parameters
        ----------
        sample : (circuit, readout_qubits, exp_readout)
            circuit - list of instructions written (gatename, qubits : tuple, parameters = None)
            readout_qubits - list of qubits that are being measured
            exp_readout - measured readout (probability generated by running same circuit multiple times on quantum computer). 
        This is in same format as readout_probs from run(). It has shape (n,2) and gives probability of finding qubit n as 0 or 1.

        Returns
        -------
        loss : jnp.array

        '''
        return self._calculate_loss(self.cross_talk_probabilities, sample[0], sample[1] )

    # def train_step(self, sample):
    #     clear_caches()
    #     loss, grad = self.grad_fun(sample[0], sample[1], self.cross_talk_probabilities )
    #     updates, self.opt_state = self.optim.update( grad , self.opt_state, value = loss )
    #     self.cross_talk_probabilities = optax.apply_updates(self.cross_talk_probabilities, updates)
    #     return loss 

    def get_config(self): #Not json serializable object, but can always pickle 
        config = {
            "num_qubits" : self.num_qubits, #int
            "num_gates" : self.num_gates, #int
            "connections" : [list(x) for x in  self.connections], #list of tuples of ints
            "readout_err" : self.readout_err.tolist(), #array -> list
            "cross_talk_probabilities" : self.cross_talk_probabilities.tolist(), #jax array
        }
        d = dict()
        for gate, val1 in self.error_operators.items():
            d[gate] = dict()
            for qubits, val2 in val1.items():
                d[gate][qubits] = dict()
                for key, val3 in val2.items():
                    d[gate][qubits][key] = val3.tolist()
        config['error_operators'] = d

        config['opt_state'] = {
            'count' : self.opt_state[0].count,
            # 'mu' : self.opt_state[0].mu.tolist(),
            # 'nu' : self.opt_state[0].nu.tolist()
        }
        return config

    def load_config(self, config):
        self.num_qubits = config['num_qubits']
        self.num_gates = config['num_gates']
        self.connections = [tuple(l) for l in config['connections']]
        self.readout_err = np.array(config['readout_err'])
        assert self.readout_err.shape == (self.num_qubits, 2)
        self.cross_talk_probabilities = jnp.array(config['cross_talk_probabilities'])
        err = config['error_operators']
        for gate, val1 in err.items():
            for qubits, val2 in val1.items():
                for key, val3 in val2.items():
                    err[gate][tuple(qubits)][key] = kraus(val3)
        self.error_operators = err

        # self.opt_state = (optax.ScaleByAdamState(
        #     count = jnp.array(config['opt_state']['count']),
            # mu = jnp.array(config['opt_state']['mu']),
            # nu = jnp.array(config['opt_state']['nu'])
        # ), optax._src.base.EmptyState())
        
    

    def __repr__(self):
        d = dict()
        for term in self.represented_terms:
            d[term] = getattr(self, term)
        return 'Model1'+ repr(d)

    def get_training_data(self, sample):
        d ={
            'scalar' : dict(),
            'image' : dict()
        }
        d['scalar']['loss'] = self.calculate_loss(sample)
        d['scalar']['log_likelihood'] = self.log_likelihood(sample)
        d['image']['cross_talk_prob'] = self.normalize_params(self.cross_talk_probabilities)



import matplotlib.pyplot as plt
def plot_cross_talk(model):
    params = model.normalize_params(model.cross_talk_probabilities)
    plt.imshow(np.log(params))
    plt.xlabel('gate being used')
    plt.ylabel('cross talk gate')
    plt.show()
    

