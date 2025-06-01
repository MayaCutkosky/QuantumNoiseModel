import pytest
from model1 import Model
from qiskit_ibm_runtime.fake_provider import FakeFez
import numpy as np

def check_eq(x,y):
    x = np.array(x)
    y = np.array(y)
    assert np.max(np.abs(x-y)) < 1e-6

def check_noisy_eq(x,y,noise = 0.3):
    x = np.array(x)
    y = np.array(y)
    assert np.max(np.abs(x-y)) < noise
    assert np.max(np.abs(x-y)) > 1e-6

@pytest.fixture
def sample_model():
    config = {
        'num_qubits' : 4,
        'connections' : ((0,1),(1,2),(2,3), (3,0)),
        'readout_err' : [[0,0] , [0,0], [0,0.1],[0.1,0.1]],
        'cross_talk_probabilities' : [ [0,0,0], [0,0,0],[0, 0, 0],[0,0,0] ],
        'opt_state' : {
            'count' : 1,
            'mu' : [.1, .2,.3],
            'nu' :[0.001, 0.004, 0.009]
        },
    }
    config['num_gates'] = config['num_qubits'] * 4 + len(config['connections'])
    
    ideal_relax = np.tile(np.identity(2), [config['num_qubits'], 1, 1, 1])
    config['error_operators'] = dict()
    for gate in ['id', 'x','sx', 'rz']:
        config['error_operators'][gate] = dict()
        for q in range(config['num_qubits']):
            config['error_operators'][gate][(q,)] = {
                'relax' : ideal_relax,
                'depol' : np.identity(2).reshape([1,2,2])
            }
    config['error_operators']['cz'] = dict()
    for qubits in config['connections']:
        config['error_operators']['cz'][qubits] = {
            'relax' : ideal_relax,
            'depol' : np.identity(4).reshape((1,4,4))
        }
    return Model(config=config)

@pytest.fixture
def sample_noisy_model():
    backend = FakeFez() 
    prop = backend.properties()
    gates = [g for g in prop.gates if np.isin(g.qubits, [0,1,2,3]).prod()]
    prop.__init__(prop.backend_name,prop.backend_version,prop.last_update_date, prop.qubits[:4], gates, None)
    return Model(prop)

#only works for sample_model1
def check_change(model):
    for gate, val1 in model.error_operators.items():
        for qubits, val2 in val1.items():
            assert val2['relax'] == np.tile(np.identity(2), [model.num_qubits, 1, 1, 1])
            if gate == 'cz':
                assert val2['depol'] == np.identity(4).reshape((1,4,4))
            else:
                assert val2['depol'] == np.identity(2).reshape((1,2,2))
    check_eq(model.readout_err, np.array([[0,0] , [0,0], [0,0.1],[0.1,0.1]]) )
        




class FakeDataset:
    def __getitem__(self,i):
        if i > 3:
            raise IndexError()
        circ = ( ('x', (0,), None),  ('x', (1,), None), ('cz', (0,1), None) )
        
        return circ, [0,1],[0,1], [0,0,0,1]





main_inst = lambda i,j : [('cz', (i,j), None)] 
h_inst = lambda j : [('sx',(j,), None),('rz', (j,), [np.pi/2]),  ('sx', (j,), None)]
x_inst = lambda i : [('x', (i,), None)]
cnot_inst =  lambda i,j : h_inst(1) + main_inst(0,1) + h_inst(1)
test1cz_exp_output = [
    (main_inst(0,1), [[1,0],[1,0]]),
    (cnot_inst(0,1), [[1,0],[1,0]]),
    (x_inst(0) + cnot_inst(0,1), [[0,1], [0,1]]),
    (x_inst(1) + cnot_inst(0,1), [[1,0],[0,1]]),
    (x_inst(0) + x_inst(1) + cnot_inst(0,1), [[0,1], [1,0]])
]

def test_model_is_not_changing(sample_model):
    circ = []
    for gate in ['id','x','sx','rz']:
        if gate == 'rz':
            param = [1.]
        else:
            param = None
        circ.append( (gate, (0,), param) )
    circ.append( ('cz', (0,1), None) ) 
    sample_model.run(circ,[0])
    check_change(sample_model)

@pytest.mark.parametrize("model, eq_fun", [('sample_model', check_eq), ('sample_noisy_model', check_noisy_eq)])
class TestRun:
        
    #testing specific gates
    @pytest.mark.parametrize('gate,exp_output',
                             [
                                 ('id',([1,0],[1,0])),
                                 ('x',([0,1],[1,0])),
                                 ('sx',([0.5,0.5],[0,1])),
                                 ('rz',([1,0],[1,0])),
                             ]
    )
    def test_1(self, model, eq_fun, gate, exp_output, request):
        model = request.getfixturevalue(model)
        if gate == 'rz':
            param = [1.]
        else:
            param = None
        circ = [(gate, (0,), param)]
        prob = model.run(circ,[0],[0])
        eq_fun(prob, [exp_output[0]])
    
        prob = model.run(circ*2,[0],[0])
        eq_fun(prob, [exp_output[1]])
    
    
    def test_1_rz(self, model, eq_fun, request):
        model = request.getfixturevalue(model)
        circ = [('sx',(0,), None), ('rz',(0,), [np.pi/2]), ('sx',(0,), None)]
        prob = model.run(circ, [0],[0])
        eq_fun(prob,[[0.5,0.5]])
    
    
    @pytest.mark.parametrize("circ, exp_output", test1cz_exp_output)
    def test_1_cz(self, model, eq_fun, circ, exp_output, request):
        model = request.getfixturevalue(model)
        prob = model.run(circ,[0,1],[0,1])
        eq_fun(prob, exp_output)

    #should work for any model
    def test_2(self, model, eq_fun, request):
        model = request.getfixturevalue(model)
        circuit =  [('x', (0,), None),('x', (1,), None)  ]
        prob =   model.run(circuit, [0,1,2],[0,1,2])
        assert prob.shape == (3,2)
        assert [abs(p[0] + p[1] - 1) < 1e-10 for p in prob]
        assert prob[0][1] > 0.9
        assert prob[1][1] > 0.9
        assert prob[2][0] > 0.9

@pytest.mark.parametrize("model_fixture, Dataset", [('sample_model', FakeDataset)])
def test_loading_from_obtained_config(model_fixture, Dataset, request):
    circ, readout_q, _ = Dataset()[0]
    model = request.getfixturevalue(model_fixture)
    config = model.get_config()
    reloaded_model = Model(config=config)
    check_eq(model.run(circ, readout_q,readout_q), reloaded_model.run(circ, readout_q, readout_q))
    
import pickle
def test_config_is_pickleable(sample_model):
    config = sample_model.get_config()
    config = pickle.loads(pickle.dumps(config))
    Model(config = config)

import jax.numpy as jnp
class TestTraining:
   def test_loss(self, sample_model):
       assert 0 == sample_model.calculate_loss(FakeDataset()[0]).round(6)
   def test_train_step(self, sample_noisy_model):
       np.random.seed(0)
       sample_noisy_model.initialize_params()
       circuits = [
               h_inst(0) + cnot_inst(1,2) + h_inst(0),
               h_inst(0) + cnot_inst(2,3) + h_inst(0),
               h_inst(2) + cnot_inst(0,1) + h_inst(2),
               h_inst(3) + cnot_inst(0,1) + h_inst(3)
           ]
       readout_qubits = [[0,1],[0,2],[1,2],[1,3]]
       outputs = []
       for circ, q in zip(circuits, readout_qubits):
           prop = sample_noisy_model.run(circ, q, [0,1,2,3])
           outputs.append(np.array([
                   prop[0][0] * prop[1][0],
                   prop[0][1] * prop[1][0],
                   prop[0][0] * prop[1][1],
                   prop[0][1] * prop[1][1]
               ]) 
           )
       true_params = np.array(sample_noisy_model.cross_talk_probabilities.tolist())
       
       sample_noisy_model.initialize_params()
       prev_dist_from_true_params = np.linalg.norm(sample_noisy_model.cross_talk_probabilities - true_params)
       for sample in zip(circuits, readout_qubits, outputs):
           loss_old = sample_noisy_model.calculate_loss(sample)
           sample_noisy_model.train_step(sample)
           loss = sample_noisy_model.calculate_loss(sample)
           curr_dist_from_true_params = np.linalg.norm(sample_noisy_model.cross_talk_probabilities - true_params)
           assert loss <= loss_old
           assert curr_dist_from_true_params <= prev_dist_from_true_params
           prev_dist_from_true_params = curr_dist_from_true_params

import tracemalloc
def test_memory_leaks(sample_model):
    sample = FakeDataset()()[0]
    tracemalloc.start()
    initial_mem, peak_mem = tracemalloc.get_traced_memory()
    for _ in range(10):
        sample_model.train_step(sample)
    final_mem = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    assert final_mem <= peak_mem
    
