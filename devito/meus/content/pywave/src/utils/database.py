import numpy as np
import segyio

class Data():
    """
    Base class to implement data bases like velocity model and density

    Parameters
    ----------
    shape : (int, ...)
        Tuple that contains the dimensions and size along each axis
    path : str
        Path to the model file
    """
    def __init__(self, shape=None, path=None):

        if shape:
            self.create_model(shape)

        if path:
            self.read_2Dmodel(path)

    def create_model(self, shape):
        self.model = np.zeros( shape, dtype=np.float32)
        self.model[:] = 1.5
 
    def read_2Dmodel(self, path):
        """
        Build a 2D velocity model from a SEG-Y format.
        It uses the 'segyio' from https://github.com/equinor/segyio
        """
        with segyio.open(path, ignore_geometry=True) as f:
            n_samples = len(f.samples)
            n_traces = len(f.trace)
            data = np.zeros(shape=(n_samples, n_traces), dtype=np.float32)
            index = 0
            for trace in f.trace:
                data[:,index] = trace
                index += 1

            # data = 2d data
            # n_samples = height
            # n_traces = length
            self.model = data


    def shape(self):
        return self.model.shape



class VelocityModel(Data):
     """
     Base class to implement the velocity model in km/s

     Parameters
     ----------
     shape : (int, ...)
         Size of the base along each axis
     path : str
         Path to the velocity model file
     """
     def __init__(self, shape=None, path=None):
         super(VelocityModel, self).__init__(shape, path)

class DensityModel(Data):
     """
     Base class to implement the density model

     Parameters
     ----------
     shape : (int, ...)
         Size of the base along each axis
     path : str
         Path to the density model file
     """
     def __init__(self, shape=None, path=None):
         super(DensityModel, self).__init__(shape, path)
