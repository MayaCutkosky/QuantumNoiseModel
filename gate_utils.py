#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 16:08:58 2025

@author: maya
"""
from jax_utils import map_qubits
from utils import JaxOperator, pauli, expi, ideal_gates
from equinox import Module
import jax
import numpy as np
ideal_gate_funs = [
    lambda params : JaxOperator(np.identity(2)),
    lambda params : JaxOperator(pauli['X']),
    lambda params : JaxOperator(np.array([[complex(1,1),complex(1,-1)],[complex(1,-1),complex(1,1)]])/2),
    lambda phi : JaxOperator([[expi(-phi[0]/2),0],[0,expi(phi[0]/2)]]),
    lambda params :(JaxOperator(pauli['I']), JaxOperator(pauli['Z'])),
]
# import qiskit_ibm_runtime
# backend = qiskit_ibm_runtime.fake_provider.FakeFez()
# prop = backend.properties()
# connections = [ tuple( g.qubits) for g in  prop.gates if g.gate == 'cz'  and g.qubits[0] < g.qubits[1] ] 
class Instruction(Module):
    gate_id : int
    qubit_ids : tuple
    params : tuple
    def __init__(self,gate_type, qubit_ids, params, num_cz_gates = 176): #hard coded value for fez!!!
        self.gate_id = list(ideal_gates.keys()).index(gate_type)
        self.qubit_ids = qubit_ids
        self.params = params
    
    def get_operator(self):
        return ideal_gate_funs[self.gate_id](self.params)
#        return jax.lax.switch(self._gate_type,  ideal_gate_funs)
        
    def get_sys_qubit_ids(self, sys_qubits):
        return map_qubits(sys_qubits, self.qubit_ids)
    def gate_type(self):
        return list(ideal_gates)[self.gate_id]



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

import numpy as np
def add_gate(circuit, used_qubits, coupling_map, distance = 1, num_gates = 1):
    used_qubits = list(used_qubits)
    possible_qubits, near_qubits = find_neighboring_qubits(used_qubits, coupling_map, distance, True)
    added_qubits = set()
    finish_loop = False
    for i in range(num_gates):
        for q1 in np.random.permutation(list(possible_qubits)):
            q1 = int(q1)
            for q2 in np.random.permutation(list(coupling_map[q1])):
                q2 = int(q2)
                if q2 not in near_qubits:
                    if q1 > q2:
                        new_gate = Instruction('cz',(q2,q1), None)
                    else:
                        new_gate = Instruction('cz', (q1,q2), None)
                    finish_loop = True
                    break
            if finish_loop:
                break
        circuit.insert(np.random.randint(len(circuit)), new_gate)
        added_qubits.add(q1)
        added_qubits.add(q2)
        
    used_qubits += list(added_qubits)
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