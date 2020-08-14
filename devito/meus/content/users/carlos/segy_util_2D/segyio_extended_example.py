import numpy as np
import sys
import matplotlib.pyplot as plt

from segyio_extended import read_segy, write_segy, duplicate_segy

# .segy files to download: http://www.agl.uh.edu/downloads/downloads.htm

def read_and_show(file_name):

    vp, _, _ = read_segy(input_file = file_name)
    plt.imshow(vp)
    plt.show()


vp1 = "/Users/hermes/Downloads/vs_marmousi-ii.segy" # unzip it first
#vp2 = "data/simplified_marmousi.segy"

read_and_show(vp1)
#read_and_show(vp2)
