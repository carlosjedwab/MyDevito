import numpy as np
import sys
import matplotlib.pyplot as plt

import unittest

from segyio_extended import read_segy, write_segy, duplicate_segy


if __name__ == '__main__':
    unittest.main()

def get_true_simple_data():
    ''' data[i][j] = j*(i+5), shape=(20,10).segy '''
    nx = 20
    ny = 10
    data = np.zeros((nx, ny))
    for i in range(0,nx):
        for j in range(0,ny):
            data[i][j] = j*(i+5)
    return data
    
class TestStringMethods(unittest.TestCase):

    def test_read(self):

        file_name = "data/simple_data_read.segy"

        data_true = get_true_simple_data()
        data_read, _, _ = read_segy(input_file = file_name)

        self.assertTrue((data_true == data_read).all())

    def test_write(self):

        file_name = "data/simple_data_write.segy"

        data_true = get_true_simple_data()
        write_segy(file_name, data_true)
        data_write, _, _ = read_segy(input_file = file_name)
        
        self.assertTrue((data_true == data_write).all())

    def test_duplicate(self):

        file_name_read = "data/simple_data_read.segy"
        file_name_write = "data/simple_data_write.segy"

        duplicate_segy(file_name_read, file_name_write)
        data_read, _, _ = read_segy(file_name_read)
        data_write, _, _ = read_segy(file_name_write)

        self.assertTrue((data_read == data_write).all())
