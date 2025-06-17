#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 17:07:13 2025

@author: maya
"""

from model1 import Model
#from qiskit_ibm_runtime import QiskitRuntimeService
from data import Dataset
import pickle
from sys import stdout
#with open('ibm_token.txt') as f:
#    ibm_token = f.read()
#service = QiskitRuntimeService(token = ibm_token, channel = 'ibm_quantum')

#backend = service.backend('ibm_fez')

dset = Dataset('Data/data_aggregationMarch6.json')

with open('checkpoint.pickle', 'rb') as f:
   chkpt = pickle.load(f)

#model = Model(backend)
model = Model(config = chkpt)
for epoch in range(100):
    for i, sample in enumerate(dset):
        if len(sample[0]) < 50 and dset.data_dicts[i]['machine'] == 'ibm_fez':
            loss = model.train_step(sample)
    with open('checkpoint.pickle', 'bw') as f:
            pickle.dump(model.get_config(), f)
    stdout.write('\r loss = %f \t\t %f  done,' % (loss, i/len(dset) ))
    stdout.flush()
    with open('loss.txt','a') as f:
        f.write(str(loss) + '\n')
        
