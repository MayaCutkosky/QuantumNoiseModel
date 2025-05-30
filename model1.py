import jax.numpy as jnp
import optax
from jax import value_and_grad
import json
from data import process_backend
import numpy as np
from utils import operator, System, ideal_gates, kraus
import pickle


class Model:
    def __init__(self, backend = None, config = None):
        self.circuit_calculation = 'only_used_qubits'
        if config is not None:
            self.load_config(config)
            self.optim = optax.adam(1e-3)
        else:
            (
                self.num_qubits,
                self.num_gates,
                self.connections,
                self.error_operators,
                self.readout_err,
            ) = process_backend(backend.properties())
            self.backend = backend
            self.initialize_params()
            
            self.optim = optax.adam(1e-3)
            self.opt_state = self.optim.init(self.cross_talk_probabilities)
        self.represented_terms = ['num_qubits', 'connections', 'error_operators', 'readout_err', 'cross_talk_probabilities']
        
        self.grad_fun = value_and_grad(self._calculate_loss, argnums = 1)
        
    def initialize_params(self):
        self.cross_talk_probabilities = np.random.rand( len(self.connections), len(self.connections)-1) / 100
        self.cross_talk_probabilities = jnp.array(self.cross_talk_probabilities.T)
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
        sys.transition_qubit(self.error_operators[gate_type][qubit_ids]['depol'] * ideal_operator, sys_qubit_ids)
        if len(qubit_ids) < 2:
            return sys
        probs = cross_talk_prob_params[self.connections.index(qubit_ids)]
        cross_talk_operator = (1 - probs.sum()) * kraus([np.identity(4)], data_object = 'jax')
        for i, (cross_talk_qubits, err_operator) in enumerate(self.error_operators[gate_type].items()):
            if cross_talk_qubits[0] in used_qubits and cross_talk_qubits[1] in used_qubits:
                p = probs[i]
                cross_talk_operator.extend(p * err_operator['depol'] * ideal_gates['cz'])
        sys.transition_qubit(cross_talk_operator, cross_talk_qubits)
        return sys
            
            

    def run(self, circuit, readout_qubits, used_qubits = None) -> np.ndarray:
        return self._run(circuit, readout_qubits, used_qubits, self.cross_talk_probabilities)
    def _run(self, circuit, readout_qubits, used_qubits, params):
        if self.circuit_calculation == 'exact':
            sys = System(self.num_qubits, data_object='jax')
        elif self.circuit_calculation == 'only_used_qubits':
            for q in list(used_qubits):
                for qn in self.backend.coupling_map.neighbors(q):
                    if qn not in used_qubits:
                        used_qubits.append(qn)

            sys = System(len(used_qubits), data_object = 'jax')


        for instruction in circuit:
            sys = self._run_instruction(sys, instruction, used_qubits, params)
        
        if self.circuit_calculation == 'only_used_qubits':
            readout_qubits =  np.array([used_qubits.index(q) for q in readout_qubits])
        readout_probs = [
            sys.rho(readout_qubits,0,0) * (1 - self.readout_err[readout_qubits,0]) + sys.rho(readout_qubits,1,1) * self.readout_err[readout_qubits,1],
            sys.rho(readout_qubits,0,0) * self.readout_err[readout_qubits,0] + sys.rho(readout_qubits,1,1) * (1 - self.readout_err[readout_qubits,1])
        ]
        readout_probs = jnp.transpose(jnp.array(readout_probs))
        return readout_probs.real

    def _calculate_loss(self, sample, params):

        circuit, readout_qubits, used_qubits, exp_readout = sample

        readout_probs = self._run(circuit, readout_qubits, used_qubits,  params)
        log_readout_probs = jnp.log(readout_probs + 1e-8) #deal with prob = 0
        binary_conversion_inds = np.arange(len(readout_qubits)), np.array([list(np.binary_repr(i, width=len(readout_qubits))) for i in range(2**len(readout_qubits))], dtype=int)
        log_pred_readout = jnp.sum(log_readout_probs[binary_conversion_inds],axis = 1)
        loss = - np.sum(jnp.array(exp_readout) * log_pred_readout)
        return loss

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
        return self._calculate_loss(sample, self.cross_talk_probabilities)

    def train_step(self, sample):
        loss, grad = self.grad_fun(sample, self.cross_talk_probabilities)
        updates, self.opt_state = self.optim.update( grad , self.opt_state )
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
            'mu' : self.opt_state[0].mu.tolist(),
            'nu' : self.opt_state[0].nu.tolist()
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

        self.opt_state = (optax.ScaleByAdamState(
            count = jnp.array(config['opt_state']['count']),
            mu = jnp.array(config['opt_state']['mu']),
            nu = jnp.array(config['opt_state']['nu'])
        ), optax._src.base.EmptyState())
        
    

    def __repr__(self):
        d = dict()
        for term in self.represented_terms:
            d[term] = getattr(self, term)
        return 'Model1'+ repr(d)

