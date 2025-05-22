import jax.numpy as jnp
import optax
from jax import grad
import json

class Model:
    def __init__(self, backend = None, config = None):
        if config is not None:
            self.num_qubits = config['num_qubits']
            self.num_gates = config['num_gates']
            self.connections = config['connections']
            self.error_operators = config['error_operators']
            self.readout_err = config['readout_err']
            self.cross_talk_probabilities = config['cross_talk_probabilities']
        else:
            (
                self.num_qubits,
                self.num_gates,
                self.connections,
                self.error_operators,
                self.readout_err,
            ) = process_backend(backend)
            self.cross_talk_probabilities = np.identity(len(self.connections) ) + np.random.rand( len(self.connections), len(self.connections)) / 100
            self.cross_talk_probabilities /= self.cross_talk_probabilities.sum(0)
            self.cross_talk_probabilities = jnp.array(self.cross_talk_probabilities.T)
            self.optim = adam(1e-3)
            self.opt_state = self.optim.init(self.cross_talk_probabilities)

    def _run_instruction(self, sys, instruction, cross_talk_prob_params):
        gate_type, qubit_ids, params = instruction
        if gate_type == 'rz':
            ideal_operator = ideal_gates[gate_type](params[0])
        else:
            ideal_operator = ideal_gates[gate_type]
        sys.transition(self.error_operators[gate_type][qubit_ids]['relax'])
        sys.transition_qubit(self.error_operators[gate_type][qubit_ids]['depol'] * ideal_operator, qubit_ids)
        if len(qubit_ids) < 2:
            return sys
        probs = cross_talk_prob_params[self.connections.index(qubit_ids)]
        
        for cross_talk_qubits, err_operator in self.error_operators[gate_type].items():
            print(cross_talk_qubits)
            p = probs[self.connections.index(cross_talk_qubits)]
            cross_talk_operator = p * err_operator['depol'] +
            (1 - p) * np.identity(4)
            kraus(cross_talk_operator) * ideal_gates['cz']
            sys.transition(cross_talk_operator, cross_talk_qubits)
        return sys

    def run(self, circuit, readout_qubits):
        return self._run(circuit, readout_qubits, self.cross_talk_probabilities)
    def _run(self, circuit, readout_qubits, params):
        sys = System(self.num_qubits)
        for instruction in circuit:
            sys = self._run_instruction(sys, instruction, params)

        readout_probs = [
            sys.rho[readout_qubits,0,0] * (1 - self.readout_err[readout_qubits,0]) + sys.rho[readout_qubits,1,1] * self.readout_err[readout_qubits,1],
            sys.rho[readout_qubits,0,0] * self.readout_err[readout_qubits,0] + sys.rho[readout_qubits,1,1] * (1 - self.readout_err[readout_qubits,1])
        ]
        readout_probs = np.transpose(readout_probs)
        return readout_probs

    def _calculate_loss(self, sample, params):
        
        circuit, readout_qubits, exp_readout = sample
        
        readout_probs = self._run(circuit, readout_qubits, params)
        log_readout_probs = np.log(readout_probs)
        log_pred_readout = np.sum(log_readout_probs[np.arange(self.num_qubits), np.array([list(np.binary_repr(i, width=self.num_qubits)) for i in range(2**self.num_qubits)], dtype=int)],axis = 0)
        loss = - exp_readout * log_pred_readout
        return loss

    def calculate_loss(self, sample):
        return self._calculate_loss(self, sample, self.cross_talk_probabilities)

    def train_step(self, sample):
        updates, self.opt_state = self.optim.update( grad(self._calculate_loss, argnums = 1)(sample, self._calculate_loss), self.opt_state )
        self.cross_talk_probabilities = optax.apply_updates(self.cross_talk_probabilities, updates)

    def get_config(self):
        config = {
            "num_qubits" : self.num_qubits,
            
        }
        d = []
        for key, val in opt_state[0]._asdict().items():
            d.append((key, val.tolist()))
        config['opt_state'] = dict(d)
}
