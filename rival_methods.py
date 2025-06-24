#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 11:45:36 2025

@author: cutkoskys
"""

import qiskit_aer
import numpy as np


def to_qiskit_circuit(circuit, readout_qubits):
    qiskit_circuit = qiskit.circuit.QuantumCircuit(num_qubits,len(readout_qubits))
    for name, qubits, params in circuit:
        if name == 'rz':
            qiskit_circuit.rz(params[0],qubits[0])
        elif name == 'cz':
            qiskit_circuit.cz(qubits[0], qubits[1])
        else:
            getattr(qiskit_circuit,name)(qubits[0])
    [qiskit_circuit.measure(q, i) for i, q in enumerate(readout_qubits)]

class Qiskit_Aer:
    def __init__(self, backend = None, properties = None):
        if properties is None:
            self.sim = qiskit_aer.AerSimulator.from_backend(backend)
        else:
            self.sim = qiskit_aer.AerSimulator(properties = properties)
    
    def test(sample):
        output = sim.run(circuit).result().data()['counts']
        return np.sum( np.log(1e-8 + sample[-1])[int(s,16) for s in outputs.keys()] * list( outputs.values()) )