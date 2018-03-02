from hrm import hrm
import pytest
from math import isclose


def test_hrm(capsys):
    test1 = hrm('test_data16.csv')
    assert isclose(test1.duration, 13.8870000839, abs_tol=10e-9)
    assert isclose(test1.mean_hr_bpm, 84.1763258787, abs_tol=10e-9)
    assert test1.num_beats == 19
    assert isclose(test1.voltage_extremes[0], -0.224999994, abs_tol=10e-9)
    assert isclose(test1.voltage_extremes[1], 0.75, abs_tol=10e-9)
    out1, err1 = capsys.readouterr()
    assert out1 == 'test_data16.json created\n'

    test2 = hrm('test_data32.csv')
    out2, err2 = capsys.readouterr()
    assert out2 == 'ECG voltage exceeds 300 mV\ntest_data32.json created\n'
