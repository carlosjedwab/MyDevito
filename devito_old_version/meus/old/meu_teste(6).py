import numpy as np
import matplotlib.pyplot as plt

########################################################################################### Create model and model0
from examples.seismic import Model, demo_model, plot_velocity, plot_perturbation

shape = (101, 101)  # Number of grid point (nx, nz)
spacing = (10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0.)  # Need origin to define relative source and receiver locations

model = demo_model('circle-isotropic',
                   vp=3.0,
                   vp_background=2.5,
                   origin=origin,
                   shape=shape,
                   spacing=spacing,
                   nbpml=40)

# For the manuscript, we'll re-form the model using the Vp field from
# this newly created model.
vp = model.vp

model0 = demo_model('circle-isotropic',
                    vp=2.5,
                    vp_background=2.5,
                    origin=origin,
                    shape=shape,
                    spacing=spacing,
                    nbpml=40)

############################################################################################

t0 = 0.     # Simulation starts a t=0
tn = 1000.  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing

nt = int(1 + (tn-t0) / dt)  # Discrete time axis length
time = np.linspace(t0, tn, nt)  # Discrete modelling time

# NOT FOR MANUSCRIPT
from devito import TimeFunction

v = TimeFunction(name="v", grid=model.grid,
                 time_order=2, space_order=4,
                 save=False)

pde = model.m * v.dt2 - v.laplace - model.damp * v.dt

# NOT FOR MANUSCRIPT
from devito import Eq
from sympy import solve

stencil_v = Eq(v.backward, solve(pde, v.backward)[0])

############################################################################################ Create sources (nshots sources), receiver (nreceivers receivers) and wave geometry

# NOT FOR MANUSCRIPT
from examples.seismic import Receiver

nshots = 21  # Number of shots to create gradient from
nreceivers = 101  # Number of receiver locations per shot 

# Recs are distributed across model, at depth of 20 m.
z_extent, _ = model.domain_size
z_locations = np.linspace(0, z_extent, num=nreceivers)
rec_coords = np.array([(980, z) for z in z_locations])

# NOT FOR MANUSCRIPT
from examples.seismic import PointSource

residual = PointSource(name='residual', ntime=nt,
                       grid=model.grid, coordinates=rec_coords)    

res_term = residual.inject(field=v.backward,
                           expr=residual * dt**2 / model.m,
                           offset=model.nbpml)

# NOT FOR MANUSCRIPT
rec = Receiver(name='rec', npoint=nreceivers, ntime=nt,
               grid=model.grid, coordinates=rec_coords)

# NOT FOR MANUSCRIPT
from examples.seismic import RickerSource

# At first, we want only a single shot.
# Src is 5% across model, at depth of 500 m.
z_locations = np.linspace(0, z_extent, num=nshots)
src_coords = np.array([(z_extent/50, z) for z in z_locations])

# NOT FOR MANUSCRIPT
f0 = 0.010  # kHz, peak frequency.
src = RickerSource(name='src', grid=model.grid, f0=f0,
                   time=time, coordinates=src_coords[nshots//2])
# NOT FOR MANUSCRIPT
plt.plot(src.time, src.data)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.show()


from examples.seismic import plot_velocity

plot_velocity(model)
plot_velocity(model0)