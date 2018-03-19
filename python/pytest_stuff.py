import pytest

def test_something():
    assert 1 == 1

if __name__ == "__main__":
    pytest.main(args=['pytest_stuff.py', '-s'])
