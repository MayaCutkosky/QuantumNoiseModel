#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 17:07:13 2025

@author: maya
"""

from model1 import Model
from qiskit_ibm_runtime import QiskitRuntimeService
from data import Dataset, restrict_machine, restrict_circuit_size
import pickle
from sys import stdout
import numpy as np
import os
def train(model, dset, epochs = 10, verbose = True, start = 0):
    def write_data(name, val):
        filename = 'Output/' + name + '.txt'
        with open(filename, 'a') as f:
            f.write( str(epoch) + ', ' + str(i) +', ' + str(val) + '\n')
    for epoch in range(start,start + epochs):
        for i, sample in enumerate(dset):
            if i > 500:
                write_data('val_log_likelihood', model.calculate_log_likelihood(sample))
                write_data('val_loss', model.calculate_loss(sample))
                continue
            
            loss = model.train_step(sample)
            
            write_data('train_loss', loss)
            if i % 10 == 0:
                write_data('train_log_likelihood',  model.calculate_log_likelihood(sample))
                np.save('Output/' + str(epoch) + '_' + str(i) + '_cross_talk_prob.npy', np.array(model.normalize_params(model.cross_talk_probabilities))  )

            stdout.write('\r epoch = %d \t loss = %f \t %f' % (epoch, loss, i/len(dset) ))
            stdout.flush()
        np.save('Output/params_chkpt', np.array(model.cross_talk_probabilities))

import jax.numpy as jnp
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file', type=str)
    parser.add_argument('--use_chkpt', action = 'store_true')
    args = parser.parse_args()
    
    with open('ibm_token.txt') as f:
        ibm_token = f.read()
    service = QiskitRuntimeService(token = ibm_token, channel = 'ibm_cloud')

    backend = service.backend('ibm_fez')
    model = Model(backend)

    if args.use_chkpt:
        model.cross_talk_probabilities = jnp.array(np.load('Output/params_chkpt.npy'))
    else:
        assert not os.path.exists('Output')
    dset = Dataset(args.dataset_file)
    dset.add_restrictions([restrict_machine('ibm_fez')])
    dset.add_restrictions([lambda x,y : len(x[0][2]) < 9 ])
    train(model, dset, start = 22, epochs = 28)
    
    

    
