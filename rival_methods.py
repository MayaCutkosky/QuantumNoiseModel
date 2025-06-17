#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 11:45:36 2025

@author: cutkoskys
"""

import qiskit_aer
import numpy as np

class Qiskit_Aer:
    def __init__(self, backend = None, properties = None):
        if properties is None:
            self.sim = qiskit_aer.AerSimulator.from_backend(backend)
        else:
            self.sim = qiskit_aer.AerSimulator(properties = properties)
    
    def test(sample):
        output = sim.run(circuit).result().data()['counts']
        return np.sum( np.log(1e-8 + sample[-1])[int(s,16) for s in outputs.keys()] * list( outputs.values()) )