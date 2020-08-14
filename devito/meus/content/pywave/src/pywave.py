from utils.compiler import Compiler
from utils.grid import Grid
from utils.database import VelocityModel, DensityModel
from utils.operator import Operator
import utils.cfl as cfl
from utils.plot import *
from utils.parser import get_options
import numpy as np

# get parsed args
args = get_options()

# verify the velocity model file
if args.vmodel:
    vel = VelocityModel(path=args.vmodel)
else:
    vel = VelocityModel(shape=tuple(args.grid))
 
# verify the density model file
if args.dmodel:
    density = DensityModel(path=args.vmodel)
else:
    density = DensityModel(shape=tuple(args.grid))

# get the velocity model dimension
dimension = len(vel.shape())

if dimension == 2:
    nz, nx = tuple(args.grid)
    dz, dx = tuple(args.spacing)
else:
    nz, nx, ny = tuple(args.grid)
    dz, dx, dy = tuple(args.spacing)

compiler = Compiler(program_version=args.tool, c_code=args.ccode)

grid = Grid(shape=vel.shape())
grid.add_source()

# apply CFL conditions
dt = cfl.calc_dt(dimension=dimension, space_order=2, spacing=tuple(args.spacing), vel_model=vel.model)
timesteps = cfl.calc_num_timesteps(args.time, dt)

params = {
    'compiler' : compiler,
    'grid' : grid,
    'vel_model' : vel,
    'density' : density,
    'timesteps': timesteps,
    'dimension' : dimension,
    'dz' : dz,
    'dx' : dx,
    #'dy' : dy,
    'dt' : dt,
    'print_steps' : args.print_steps
}

op = Operator(params)

wavefield, exec_time = op.forward()

print("dt: %f miliseconds" % dt)
print("Number of timesteps:", timesteps)
print("Forward execution time: %f seconds" % exec_time)

plot(wavefield)
show(vel.model)

print(vel.shape())
print(vel.model)
