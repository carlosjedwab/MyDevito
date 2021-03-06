import numpy as np
import matplotlib as plt

#############################################################################################

#NBVAL_IGNORE_OUTPUT
from examples.seismic import Model, plot_velocity
a = 2**12
# Define a physical size
shape = (a, a)  # Number of grid point (nx, nz)
spacing = (10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0.)  # What is the location of the top left corner. This is necessary to define
# the absolute location of the source and receivers

# Define a velocity profile. The velocity is in km/s
v = np.empty(shape, dtype=np.float32)
v[:, :51] = 1.5
v[:, 51:] = 2.5

# With the velocity and model size defined, we can create the seismic model that
# encapsulates this properties. We also define the size of the absorbing layer as 10 grid points
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
              space_order=2, nbpml=10)


############################################################################################

from examples.seismic import TimeAxis

t0 = 0.  # Simulation starts a t=0
tn = 1000.  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing

time_range = TimeAxis(start=t0, stop=tn, step=dt)

##############################################################################################

#NBVAL_IGNORE_OUTPUT
from examples.seismic import RickerSource

f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=1, time_range=time_range)

# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = np.array(model.domain_size) * .5
src.coordinates.data[0, -1] = 20.  # Depth is 20m



#############################################################################################

#NBVAL_IGNORE_OUTPUT
from examples.seismic import Receiver

# Create symbol for a receivers
rec = Receiver(name='rec', grid=model.grid, npoint=a, time_range=time_range)

# Prescribe even spacing for receivers along the x-axis
rec.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=a)
rec.coordinates.data[:, 1] = 20.  # Depth is 20m


###################################################################################################

# In order to represent the wavefield u and the square slowness we need symbolic objects 
# corresponding to time-space-varying field (u, TimeFunction) and 
# space-varying field (m, Function)
from devito import TimeFunction

# Define the wavefield with the size of the model and the time dimension
u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2)

# We can now write the PDE
pde = model.m * u.dt2 - u.laplace + model.damp * u.dt

# The PDE representation is as on paper
pde

#print(pde)

#####################################################################################################

# This discrete PDE can be solved in a time-marching way updating u(t+dt) from the previous time step
# Devito as a shortcut for u(t+dt) which is u.forward. We can then rewrite the PDE as 
# a time marching updating equation known as a stencil using customized SymPy functions
from devito import Eq, solve

stencil = Eq(u.forward, solve(pde, u.forward))

#print(u.forward)

#####################################################################################################


# Finally we define the source injection and receiver read function to generate the corresponding code
src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)

# Create interpolation expression for receivers
rec_term = rec.interpolate(expr=u.forward)

##########################################################################################################

from devito import configuration
configuration['openmp'] = True

#NBVAL_IGNORE_OUTPUT
from devito import Operator

save = 1

if save == 1:
    op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)
    text_file = open("operator4.h", "w")
    text_file.write(str(op.ccode))
    text_file.close()
else:
    code_file = open("operator4.h","r")
    code = code_file.read()

    op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map, code = code)
    #op2 = Operator([Eq(u.forward,u)], code)

    #op(dt=model.critical_dt)
    op(time=time_range.num-1, dt=model.critical_dt) 

    from examples.seismic import plot_shotrecord

    #plot_shotrecord(rec.data, model, t0, tn)