import pytest
from data import process_backend
from qiskit_ibm_runtime.fake_provider import FakeFez

@pytest.fixture
def backend_data():
    backend = FakeFez() 
    return process_backend(backend.properties())


