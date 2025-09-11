import numpy as np

#load model


def eval_removing_circ(dset, model, verbose = False):
    dset.add_restrictions( lambda sample, data_dict : len(sample[0][2]) < 10 and data_dict['Neighbor'] is not None)
    total_right = 0
    for j, sample in enumerate(dset):
        true_loglikelihood = model.calculate_log_likelihood(sample)
        circ = sample[0][0]
        circ_fake = []
        for i in circ:
            if np.all( np.isin( i.qubit_ids, sample[0][1] )):
                circ_fake.append(i)
        
        sample_fake = ( (circ_fake, sample[0][1], sample[0][2]) , sample[1])
        fake_loglikelihood = model.calculate_log_likelihood(sample_fake)
        
        total_right += ( fake_loglikelihood < true_loglikelihood )
        print(total_right / (j+1) )
    
    return total_right / len(dset)
    
