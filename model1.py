import jax.numpy as jnp
import optax
from jax import value_and_grad, clear_caches, random, custom_gradient
import json
from data import process_backend
import numpy as np
from utils import System, ideal_gates, kraus, DensityMatrix, softmax, pauli, operator
import pickle
import itertools as it

#from cursor
@custom_gradient
def ste_choice(p, key, n):
    # Forward: discrete sampling
    idx = random.choice(key, jnp.arange(n), p=p)
    
    # Backward: gradient flows through p
    def grad_fn(dy):
        return dy * jnp.ones_like(p), None, None
    
    return idx, grad_fn

class RandomGenerator:
    def __init__(self):
        self.key = random.PRNGKey(0)
        self.i = 0
    def get_subkey(self):
        self.i += 1
        return random.fold_in(self.key, self.i)
        
    def random_choice(self, p, n):
        subkey = self.get_subkey()
        return ste_choice(p, subkey,n)

def compute_prob_matrix(M): #from cursor
    n = M.shape[0]
    size = 2 ** n

    # Generate all possible n-bit vectors for A and B
    A_bits = ((np.arange(size)[:, None] >> np.arange(n-1, -1, -1)) & 1).astype(np.uint8)  # (size, n)
    B_bits = ((np.arange(size)[:, None] >> np.arange(n-1, -1, -1)) & 1).astype(np.uint8)  # (size, n)

    # Expand A_bits and B_bits to shape (size, size, n)
    A_expand = A_bits[:, None, :]  # (size, 1, n)
    B_expand = B_bits[None, :, :]  # (1, size, n)

    # Compute probabilities for each digit position
    mask_A1 = (A_expand == 1)
#    mask_A0 = ~mask_A1

    prob_A1 = np.where(B_expand == 0, M[:, 0], 1 - M[:, 0])  # (1, size, n)
    prob_A0 = np.where(B_expand == 1, M[:, 1], 1 - M[:, 1])  # (1, size, n)

    probs = np.where(mask_A1, prob_A1, prob_A0)  # (size, size, n)

    M_prime = np.prod(probs, axis=2)  # (size, size)

    return jnp.array(M_prime)


class Model:
    def __init__(self, backend = None, backend_properties = None, config = None):
        self.circuit_calculation = 'only_used_qubits'
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
            self.backend = backend
            self.initialize_params()
            
            self.optim = optax.polyak_sgd(0.001, 
                                          f_min = 0, 
                                          eps = 1e-8, 
                                          scaling= optax.schedules.linear_schedule(1, 1e-5, 1e4)
            )

            self.opt_state = self.optim.init(self.cross_talk_probabilities)        
        self.represented_terms = ['num_qubits', 'connections', 'cross_talk_probabilities']
        
        self.regularization_fun = lambda p: 0#jnp.var(p,axis=0).sum() 
        self.grad_fun = value_and_grad(self._calculate_loss, argnums = 1)
        
        self.coupling_map = dict()
        for q1, q2 in self.connections:
            self.coupling_map.setdefault(q1, set() )
            self.coupling_map.setdefault(q2, set() )
            self.coupling_map[q2].add(q1)
            self.coupling_map[q1].add(q2)

    def initialize_params(self, noise = 0.01):
        r = np.random.rand( len(self.connections), len(self.connections)) / self.num_qubits * noise
        r[np.arange(len(self.connections)), np.arange(len(self.connections))] = 0        
        self.cross_talk_probabilities = np.diag(1 - r.sum(axis = 1)) + r
        self.cross_talk_probabilities = jnp.array(self.cross_talk_probabilities)
    
    def transition_depol_err(self,sys, gate_type, qubit_ids, sys_qubit_ids):
        depol_err = self.error_operators[gate_type][qubit_ids]['depol']
        if random.uniform(self.random_gen.get_subkey()) > depol_err:
            return #no error
        depol_operator = self.random_gen.choice(self.random_gen.get_subkey(), list(it.combinations(pauli, len(sys_qubit_ids)))[1:])
        sys.transition_qubit(depol_operator, sys_qubit_ids)

        
    def _run_instruction(self, sys, instruction, used_qubits, cross_talk_prob_params):
        gate_type, qubit_ids, params = instruction
        if self.circuit_calculation == 'only_used_qubits':
            sys_qubit_ids = []
            for q in qubit_ids:
                sys_qubit_ids.append(used_qubits.index(q))
        else:
            sys_qubit_ids = qubit_ids
        if gate_type == 'rz':
            ideal_operator = ideal_gates[gate_type](params[0])
        else:
            ideal_operator = ideal_gates[gate_type]
        
        sys.transition(self.error_operators[gate_type][qubit_ids]['relax'][used_qubits])
        sys.transition_qubit(ideal_operator, sys_qubit_ids)
        self.transition_depol_err(sys, gate_type, qubit_ids, sys_qubit_ids)

        if len(qubit_ids) < 2:
            return sys
        
        #$\\rho^{(1)} = (1 - \\sum_{g' \\in G} p_{g,g'}) C_g R_T \\rho^{(0)} R_T^\\dagger C_g^{\\dagger} + \\sum_{g' \\in G}  p_{g,g'} C_{g'}C_g R_T \\rho^{(0)} R_T^\\dagger (C_gC_{g'})^{\\dagger}  $
        
        #probabilitic method
        ind = np.isin(self.connections,used_qubits).prod(1)
        connections =  [qubits for (i, qubits) in zip(ind, self.connections) if i]
        probs = cross_talk_prob_params[connections.index(qubit_ids)]
        i = self.random_gen.random_choice(probs, len(connections) )
        cross_talk_qubits = connections[i]
        if cross_talk_qubits != qubit_ids:
            sys_cross_talk_qubit_ids =  [used_qubits.index(q)  for q in cross_talk_qubits]
            sys.transition_qubit(ideal_gates['cz'],sys_cross_talk_qubit_ids)            
            self.transition_depol_err(sys, gate_type, qubit_ids, sys_cross_talk_qubit_ids)
        return sys
            
            

    def run(self, circuit, readout_qubits, used_qubits = None) -> np.ndarray:
        return self._run(circuit, readout_qubits, used_qubits, self.prepare_params(self.cross_talk_probabilities, used_qubits))
    def _run(self, circuit, readout_qubits, used_qubits, params):
        
        binary_conversion_inds = np.array([list(np.binary_repr(i, width=len(readout_qubits))) for i in range(2**len(readout_qubits))], dtype=int)
        output= 0
        
        readout_err_mat = compute_prob_matrix(self.readout_err[readout_qubits])
        readout_qubits =  [used_qubits.index(q) for q in readout_qubits]
        
        self.random_gen = RandomGenerator()
        
        for t in range(10):
            sys = System(len(used_qubits), data_object = 'jax')
            for j, instruction in enumerate(circuit):
                sys = self._run_instruction(sys, instruction, used_qubits, params)
                if j % 100 == 0:
                    readout_probs = sys.calc_probabilities(readout_qubits)
                    print(np.max(readout_probs))                    
            # assert (jnp.isfinite(sys.rho._data) ).prod()
            readout_probs = sys.calc_probabilities(readout_qubits)
            print(np.max(readout_probs))
            
            readout_probs = jnp.matmul(readout_probs, readout_err_mat)
            output += readout_probs/10
        return output
        

    @staticmethod
    def normalize_params(params):
        exp_x = jnp.exp(params).T
        return (exp_x / exp_x.sum(0)).T
    
    def prepare_params(self, params, used_qubits):
        ind = np.where(np.isin(self.connections,used_qubits).prod(1))[0]
        return self.normalize_params(params)[ind][:,ind]
        
    
    
    def calculate_log_likelihood(self,sample):
        (circuit, readout_qubits, used_qubits), exp_readout = sample
        log_pred_readout = self._run(circuit, readout_qubits, used_qubits, self.normalize_params(self.cross_talk_probabilities))
        return np.sum(np.array(exp_readout) * log_pred_readout)
 
        
    def _calculate_loss(self, sample, params):

        (circuit, readout_qubits, used_qubits), exp_readout = sample
        
        params = self.prepare_params(params, used_qubits)
        
        
        non_zero_inds = np.nonzero(exp_readout)[0]
        readout_probs = self._run(circuit, readout_qubits, used_qubits,  params)
        
        log_pred_readout = jnp.log(readout_probs[non_zero_inds] + 1e-9) #deal with prob = 0    
        log_exp_readout = jnp.log((exp_readout/exp_readout.sum())[non_zero_inds])
        loss =  jnp.sum( jnp.square(log_exp_readout - log_pred_readout) )
        cross_talk_probs = params.at[np.arange(len(params)), np.arange(len(params))].multiply(0)#.subtract(params[np.arange(len(params)), np.arange(len(params))])
        
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
        return self._calculate_loss(sample, self.normalize_params(self.cross_talk_probabilities) )

    def train_step(self, sample):
        clear_caches()
        loss, grad = self.grad_fun(sample, self.cross_talk_probabilities )
        assert np.isfinite(grad).prod()
        updates, self.opt_state = self.optim.update( grad , self.opt_state, value = loss )
        self.cross_talk_probabilities = optax.apply_updates(self.cross_talk_probabilities, updates)
        return loss 

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
    
