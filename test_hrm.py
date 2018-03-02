from hrm import hrm
import pytest


def test_hrm(capsys):
    test1 = hrm('test_data16.csv')
    assert test1.beats == [0.0430000015, 0.7929999828, 1.5429999828
        , 2.2929999828, 3.0429999828, 3.7929999828, 4.5430002213
        , 5.2930002213, 6.0430002213, 6.7930002213, 7.5430002213
        , 8.2930002213, 9.0430002213, 9.7930002213, 10.5430002213
        , 11.2930002213, 12.0430002213, 12.7930002213, 13.5430002213]
    assert test1.duration == 13.8870000839
    assert test1.mean_hr_bpm == 84.1763258787
    assert test1.num_beats == 19
    assert test1.voltage_extremes == [-0.224999994, 0.75]
    out1, err1 = capsys.readouterr()
    assert out1 == 'test_data16.json created'

    test2 = hrm('test_data32.csv')
    out2, err2 = capsys.readouterr()
    assert out2 == 'ECG voltage greater than 300mV'
