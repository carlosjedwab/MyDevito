import segyio
import numpy as np

def read_segy(input_file):
    ''' Reads a 2d data from disk in a segy format '''

    with segyio.open(input_file, ignore_geometry=True) as f:
        n_samples = len(f.samples)
        n_traces = len(f.trace)
        data = np.zeros(shape=(n_samples, n_traces))
        index = 0
        for trace in f.trace:
            data[:,index] = trace
            index += 1
    
    # data = 2d data
    # n_samples = height
    # n_traces = length
    return data, n_samples, n_traces


def write_segy(output_file, data):
    ''' Writes a 2d data into disk as a segy format '''

    # empirical adjustment results
    data = np.rot90(data)
    data = np.flip(data, 0)
    data = np.ascontiguousarray(data)

    traces = np.array(data)

    n_samples = data.shape[0]
    n_traces = data.shape[1]

    spec = segyio.spec()
    spec.sorting = 2
    spec.format = 1
    spec.samples = range(n_traces)
    spec.ilines = range(n_samples)
    spec.xlines = range(1) # for 2d this is always 1, otherwise it's the number of transects

    with segyio.create(output_file, spec) as dst:
        dst.header[:] = []
        dst.trace[:] = np.float32(traces[:])


def duplicate_segy(input_file, output_file):
    ''' Duplicates a 2d data from disk into disk as a segy format '''

    with segyio.open(input_file, ignore_geometry=True) as src:
        spec = segyio.tools.metadata(src)
        with segyio.create(output_file, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            dst.header = src.header
            dst.trace = src.trace


import matplotlib as plt

file_name = "data/Simple2D/vp_marmousi_bi"
vp, _, _ = read_segy(input_file = file_name)
plt.imshow(vp)
plt.show()