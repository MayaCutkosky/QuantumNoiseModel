#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 11:59:26 2025

@author: maya
"""

import qiskit_ibm_runtime
backend = qiskit_ibm_runtime.fake_provider.FakeFez()
from model1 import Model
model = Model(backend)
from data import Dataset, restrict_machine, restrict_circuit_size
dset = Dataset('../Downloads/QuantumCrosstalkData/data_aggregationMarch6.json')
sample = dset[60]
import equinox as eqx
import numpy as np
print( model.calculate_loss(sample) )





