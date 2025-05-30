import pytest
from data import process_backend
from qiskit_ibm_runtime.fake_provider import FakeFez
from numpy import ndarray
from utils import Kraus

def test_process_backend():
    backend = FakeFez() 
    output = process_backend(backend.properties())
    assert isinstance(output[0], int)
    assert isinstance(output[1], int)
    assert isinstance(output[3], dict)
    assert isinstance(output[4], ndarray)


