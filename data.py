from qiskit.quantum_info import average_gate_fidelity, Choi
from qiskit.quantum_info import Kraus as qiskit_Kraus
import itertools as it
import numpy as np
import warnings
from utils import pauli, operator, kraus

def relaxation_error(prop, qubit, gate_length):
    t1,t2 = prop.t1(qubit), prop.t2(qubit)
    if t1 * 2 < t2:
        warnings.warn("t2 > 2t1 for qubit {}. Changing {} to {}".format(qubit, t2, 2 * t1))
        t2 = 2 * t1

    p_reset = 1 - np.exp(-gate_length /t1)
    exp_t2 = np.exp(-gate_length / t2)
    if p_reset - 1 < 1e-12 or exp_t2 < 1e-12:
        return [np.identity(2)]
    err = qiskit_Kraus(Choi(
                np.array(
                    [
                        [1, 0, 0, exp_t2],
                        [0, 0, 0, 0],
                        [0, 0, p_reset, 0],
                        [exp_t2, 0, 0, 1 - p_reset],
                    ]
                )
            ))
    err = err.data
    err.extend([np.identity(2)]*(3-len(err)))
    return err

def gate_error(gate_err, relax_err, num_qubits):
    relax_fid = average_gate_fidelity(relax_err)
    dim = 2 ** num_qubits
    depol_param = dim * (gate_err + relax_fid - 1) / (dim * relax_fid - 1)
    return depol_param / 4 ** num_qubits

def process_gate_data(g):
  gate_length = None
  gate_error = None
  for param in g.parameters:
      if param.name == 'gate_length':
        gate_length = param.value
        if param.unit == 'ns':
          gate_length *= 1e-9
      if param.name == 'gate_error':
        gate_error = param.value
  return g.gate, tuple(g.qubits), gate_error, gate_length

def get_errs_from_gate(prop, g):
    name, qubits, gate_err, gate_length = process_gate_data(g)
    relax_errs = []
    for q in range(len(prop.qubits)):
        relax_errs.append(relaxation_error(prop, q, gate_length))

    relax_gate_err = qiskit_Kraus([1])
    for q in qubits:
        relax_gate_err = relax_gate_err.expand(qiskit_Kraus(relax_errs[q]))
    depol_err = gate_error(gate_err, relax_gate_err, len(qubits))
    if depol_err < 0 :
        return kraus(relax_errs), operator(np.identity(2**len(qubits)))
    gen = it.product(pauli.keys(),repeat = len(qubits))
    gen.__next__()
    kraus_input = []
    for key_list in gen:
        pauli_err = operator([1])
        for key in key_list:
            pauli_err = pauli_err.tensor(pauli[key])
        kraus_input.append(depol_err * pauli_err)
    kraus_input.append((1-len(kraus_input)*depol_err) * np.identity(2**len(qubits)))
    return kraus(relax_errs), kraus(kraus_input)

def get_readout_errs(q):
    probs = [None, None]
    for param in q:
        if param.name == 'prob_meas0_prep1':
            probs[0] = param.value
        elif param.name == 'prob_meas1_prep0':
            probs[1] = param.value
    return probs

def process_backend(prop):
    errs = dict()
    for name in ['id','cz','x','sx','rz']:
        errs[name] = dict()
    connections = []
    for g in prop.gates:
        name, qubits, gate_err, gate_length = process_gate_data(g)
        if name == 'cz':
            if qubits[0] > qubits[1]: #risky, might undercount if missunderstand how ibm is recording. Only works for cz and other twoway gates
                continue
            connections.append(qubits)
        if name in ['id','cz', 'x','sx', 'rz']:
            relax_errs, depol_err = get_errs_from_gate(prop,g)
            errs[name][qubits] = {
                'relax' : relax_errs,
                'depol' : depol_err
            }
    readout_errs = np.empty([len(prop.qubits), 2])
    for i,q in enumerate(prop.qubits):
        readout_errs[i] = get_readout_errs(q)
    return len(prop.qubits), len(prop.gates), connections, errs, readout_errs


import json
import qiskit
class Dataset:
    def __init__(self, filename):
        self.data_dir = filename.rpartition('/')[0]
        with open(filename) as f:
            self.data_dicts = json.load( f)
        

    def load_circuit(self, filename):
        with open(self.data_dir + '/TranspiledCircuits/' + filename, 'br') as fd:
            circuits = qiskit.qpy.load(fd)
            return circuits
    @staticmethod
    def translate_circuit(circuit):
        circ = []
        measurement_gate_inds = dict()
        used_qubits = set()
        for i, inst in enumerate(circuit):
            qubits = tuple([q._index for q in inst.qubits])
            for q in qubits:
                used_qubits.add(q)
            if inst.name == 'measure':
                measurement_gate_inds[inst.clbits[0]._index] = inst.qubits[0]._index
            if inst.name in ['barrier','measure']:
                continue
            if len(qubits) == 2:
                if qubits[0] > qubits[1]:
                    qubits = (qubits[1], qubits[0])
            circ.append([inst.name, qubits, inst.params])
        readout_qubits = np.empty(len(measurement_gate_inds), dtype = int)
        for ind, val in measurement_gate_inds.items():
            readout_qubits[ind] = val
        return circ, readout_qubits, list(used_qubits)
    
    @staticmethod
    def translate_job_measurements(job_measurements, num_readout_qubits):
        readout_values = [ int(i, 2) for i in job_measurements.keys() ]
        exp_readout = np.zeros(2**num_readout_qubits)
        exp_readout[readout_values] = list(job_measurements.values())
        return exp_readout
    
    def __getitem__(self, i):
        d = self.data_dicts[i]
        #somehow load data
        circuit, readout_qubits, used_qubits =  self.translate_circuit(self.load_circuit(d['filename'])[0])
        exp_readout = self.translate_job_measurements(d['job_measurements'], len(readout_qubits))
        
        return circuit, readout_qubits, used_qubits, exp_readout
        
    def __len__(self):
        return len(self.data_dict)




