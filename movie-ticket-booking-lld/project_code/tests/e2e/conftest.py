import pytest
import requests
import time

BASE_URL = "http://localhost:8080/api"

@pytest.fixture(scope="function")
def api_session():
    # Attempt to reset DB before each test
    try:
        requests.post(f"{BASE_URL}/test/reset", timeout=5)
    except requests.exceptions.RequestException:
        pass # Ignore if backend not up yet
    
    session = requests.Session()
    yield session
    
    # Optionally reset after
    try:
        requests.post(f"{BASE_URL}/test/reset", timeout=5)
    except requests.exceptions.RequestException:
        pass
