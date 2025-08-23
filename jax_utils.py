#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 09:21:25 2025

@author: maya
"""
import jax.numpy as jnp
#from chatgpt
def map_qubits(used_qubits, readout_qubits):
    used_qubits = jnp.array(used_qubits)  # shape (m,)
    readout_qubits = jnp.array(readout_qubits)  # shape (n,)

    # Compare all pairs (n, m)
    matches = readout_qubits[:, None] == used_qubits[None, :]
    # Argmax along axis 1 gives the index in used_qubits
    return jnp.argmax(matches, axis=1)

#also from chatgpt
def make_inverse_map(sys_qubits):
    max_id = int(jnp.max(sys_qubits))
    inv_map = -jnp.ones(max_id + 1, dtype=jnp.int32)
    inv_map = inv_map.at[sys_qubits].set(jnp.arange(len(sys_qubits)))
    return inv_map