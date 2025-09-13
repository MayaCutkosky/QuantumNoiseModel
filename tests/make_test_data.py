#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 09:29:30 2025

@author: maya
"""

import qiskit_aer
import numpy as np
from gate_utils import Instruction
from utils import ideal_gates
gate_names = list(ideal_gates.keys())
#sim = qiskit_aer.AerSimulator()
class Dataset:
    def __init__(self, connections, read_all_qubits = True, used_qubit_size= None, circuit_size = None):
        self.read_all_qubits = read_all_qubits
        self.used_qubit_size = used_qubit_size
        self.circuit_size = circuit_size
        self.connections = connections
    def exp_readout_fun(self, inp, rand_gen):
        rand_out = np.round(rand_gen.random(2 ** len(inp[1])) * 1024).astype(int)
        return rand_out
    def __getitem__(self, i):
        if i >= 100:
            raise IndexError("Index out of bounds")
        gen = np.random.default_rng(i)
        size_used_qubits = gen.integers(2,self.used_qubit_size+1 )
        if self.read_all_qubits:
            size_readout_qubits = size_used_qubits
        else:            
            size_readout_qubits = gen.integers(2,size_used_qubits)
        used_qubits = gen.choice( np.arange(156), size_used_qubits, replace = False)
        
        readout_qubits = gen.choice( used_qubits, size_readout_qubits, replace = False)
        circuit = []
        for j in range( self.circuit_size):
            
            gate_type = gen.choice(gate_names)
            
            if gate_type == 'rz':
                params =  ( (gen.random() - 0.5) * np.pi,)
            else:
                params = ()
            
            if gate_type == 'cz':
                qubit_ids = gen.choice(self.connections)
            else:
                qubit_ids = gen.choice(used_qubits, 1)
            
            circuit.append( Instruction( gate_type, qubit_ids, params ) )

        inp = (circuit, readout_qubits, used_qubits)
        return inp, self.exp_readout_fun(inp, gen)