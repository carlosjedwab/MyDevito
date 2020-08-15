#NBVAL_IGNORE_OUTPUT
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

# This cell sets up the problem that is already explained in the first TLE tutorial.

from examples.seismic import Receiver
from examples.seismic import RickerSource
from examples.seismic import Model, plot_velocity, TimeAxis
from devito import TimeFunction
from devito import Eq, solve
from devito import Operator


# Set velocity model
nx = 201
nz = 201
nb = 10
shape = (nx, nz)
spacing = (20., 20.)
origin = (0., 0.)
v = np.empty(shape, dtype=np.float32)
v[:, :int(nx/2)] = 2.0
v[:, int(nx/2):] = 2.5

model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
              space_order=2, nbpml=10)

# Set time range, source, source coordinates and receiver coordinates
t0 = 0.  # Simulation starts a t=0
tn = 10000.  # Simulation lasts tn milliseconds
dt = model.critical_dt  # Time step from model grid spacing
time_range = TimeAxis(start=t0, stop=tn, step=dt)
nt = time_range.num  # number of time steps

f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(
    name='src',
    grid=model.grid,
    f0=f0,
    time_range=time_range)  

src.coordinates.data[0, :] = np.array(model.domain_size) * .5
src.coordinates.data[0, -1] = 20.  # Depth is 20m

rec = Receiver(
    name='rec',
    grid=model.grid,
    npoint=101,
    time_range=time_range)  # new
rec.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=101)
rec.coordinates.data[:, 1] = 20.  # Depth is 20m
depth = rec.coordinates.data[:, 1]  # Depth is 20m


plot_velocity(model, source=src.coordinates.data,
              receiver=rec.coordinates.data[::4, :])

#Used for reshaping
vnx = nx+20 
vnz = nz+20

# Set symbolics for the wavefield object `u`, setting save on all time steps 
# (which can occupy a lot of memory), to later collect snapshots (naive method):

u = TimeFunction(name="u", grid=model.grid, time_order=2,
                 space_order=2, save=time_range.num)

# Set symbolics of the operator, source and receivers:
pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
stencil = Eq(u.forward, solve(pde, u.forward))
src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m,
                      offset=model.nbpml)
rec_term = rec.interpolate(expr=u, offset=model.nbpml)
op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)

# Run the operator for `(nt-2)` time steps:
op(time=nt-2, dt=model.critical_dt)


#########################################################################################################

#NBVAL_IGNORE_OUTPUT
from devito import ConditionalDimension

nsnaps = 100               # desired number of equally spaced snaps
factor = round(nt / nsnaps)  # subsequent calculated factor

print(f"factor is {factor}")

#Part 1 #############
time_subsampled = ConditionalDimension(
    't_sub', parent=model.grid.time_dim, factor=factor)
usave = TimeFunction(name='usave', grid=model.grid, time_order=2, space_order=2,
                     save=(nt + factor - 1) // factor, time_dim=time_subsampled)
print(time_subsampled)
#####################

u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2)
pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
stencil = Eq(u.forward, solve(pde, u.forward))
src_term = src.inject(
    field=u.forward,
    expr=src * dt**2 / model.m,
    offset=model.nbpml)
rec_term = rec.interpolate(expr=u, offset=model.nbpml)

#Part 2 #############
op1 = Operator([stencil] + src_term + rec_term,
               subs=model.spacing_map)  # usual operator
op2 = Operator([stencil] + src_term + [Eq(usave, u)] + rec_term,
               subs=model.spacing_map)  # operator with snapshots

op1(time=nt - 2, dt=model.critical_dt)  # run only for comparison
op2(time=nt - 2, dt=model.critical_dt)
#####################

#Part 3 #############
print("Saving snaps file")
print("Dimensions: nz = {:d}, nx = {:d}".format(nz + 2 * nb, nx + 2 * nb))
filename = "snaps2.bin"
usave.data.tofile(filename)
#####################

#NBVAL_IGNORE_OUTPUT
fobj = open("snaps2.bin", "rb")
snaps = np.fromfile(fobj, dtype=np.float32)
snaps = np.reshape(snaps, (nsnaps, vnx, vnz))
fobj.close()

plt.rcParams['figure.figsize'] = (20, 20)  # Increases figure size

imcnt = 1 # Image counter for plotting
plot_num = 10 # Number of images to plot

ims = []
for i in range(0, nsnaps, int(nsnaps/plot_num)):
   fig = plt.figure()
   imcnt = imcnt + 1
   plt.imshow(np.transpose(snaps[i,:,:]))
   plt.savefig('grafico_teste/grafico_onda_num_%d.png' % (imcnt))
