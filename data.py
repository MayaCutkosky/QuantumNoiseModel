from qiskit.quantum_info import average_gate_fidelity, Choi
from qiskit.quantum_info import Kraus as qiskit_Kraus

import numpy as np
import warnings
def relaxation_error(prop, qubit, gate_length):
    t1,t2 = prop.t1(qubit), prop.t2(qubit)
    if t1 * 2 < t2:
        warnings.warn("t2 > 2t1 for qubit {}. Changing {} to {}".format(qubit, t2, 2 * t1))
        t2 = 2 * t1

    p_reset = 1 - np.exp(-gate_length /t1)
    exp_t2 = np.exp(-gate_length / t2)
    if p_reset - 1 < 1e-12 and exp_t2 < 1e-12:
        return np.identity(2)
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

def process_backend(prop):
    errs = dict()
    for name in ['id','cz','x','sx','rz']:
        errs[name] = dict()
    connections = []
    for g in prop.gates:
        name, qubits, gate_err, gate_length = process_gate_data(g)
        if name == 'cz':
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

import zipfile
f = zipfile.ZipFile('QuantumCrosstalkData.zip')
import json
d = json.loads( f.read('data_aggregationMarch6.json') )

def load_circuit(filename):
  with f.open('TranspiledCircuits/' + filename, 'r') as fd:
      circuits = qiskit.qpy.load(fd)
  return circuits

def translate_circuit(circuit):
    return [ [inst.name, [q._index for q in inst.qubits], inst.params] for inst in circuit.data]


