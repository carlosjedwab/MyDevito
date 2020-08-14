# Two-Dimensional Finite-Difference Wave Propagation in Isotropic Media (ISO2D)

This baseline is a pure C version of the simple acoustic wave model available [here](https://software.intel.com/en-us/articles/code-sample-two-dimensional-finite-difference-wave-propagation-in-isotropic-media-iso2dfd)

This code has two versions: 

* iso2d-array2d.c - This implementation represents the grid as a 2 dimentional array[N][M]
* iso2d.c - This implementation represents the grid as a 1 dimensional array[N*M]

To compile and run the code:

```
# compile using the Makefile
make

# run the program
./binfile N M ITERATIONS

- binfile: compiled file
- N: rows of the grid
- M: cols of the grid
- ITERATIONS: number of iterations or timesteps

# run the version with array[N*M]
./iso2d 512 512 3000

# or run the version with array[N][M]
./iso2d-array2d 512 512 3000

```

Once you run the code, the result file will be saved in the `wavefield` directory.

You can plot the result using the `plot.py` script. The result plot will be saved in the `plots` directory.


```
python3 plot.py --file <path_to_file> --name <name_to_the_image>

python3 plot.py --file wavefield/wavefield-iter-3000-grid-512-512.txt --name figure

```

You also can generate more wavefield files over the iterations and create a video showing the wave propagation. 
In order to do that, uncomment the lines 141 and 142 (in iso2d.c) or 129 e 130 (in iso2d-array2d.c), run the code and execute the `movie.py` script.


```
python3 movie.py --wavefields <folder_with_wavefield_files> --iterations <iterations> --n1 <grid_size_rows> --n2 <grid_size_cols> --hop <number_of_hops> --name <video_name>

python3 movie.py --wavefields wavefield/ --iterations 3000 --n1 512 --n2 512 --hop 10 --name wave-propagation

```

You might need to install a few python packages to run the python scripts:


`pip3 install numpy matplotlib argparse opencv-python`
