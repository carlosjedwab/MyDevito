import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sympy import init_printing, pprint, expand, latex, simplify, factor, symbols

from examples.seismic import Model, plot_velocity
from examples.seismic import TimeAxis
from examples.seismic import RickerSource
from examples.seismic import Receiver
from examples.seismic import plot_shotrecord, plot_image
from examples.cfd import init_smooth, plot_field

from devito import Eq, solve
from devito import TimeFunction
from devito import Operator
from devito import configuration

from matplotlib import cm

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

from time import time
import os


tntotal = np.arange(10,900,30)   ### YOU CAN CHANGE THE TIMESTEPS
start_time = time()
try:
	os.mkdir('devito_acoustic_wave')
except:
	pass
ims = []
fig = plt.figure()
for tn in tntotal:
	# Define a physical size
	shape = (101, 101)    # Number of grid point (nx, nz)
	spacing = (10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
	origin = (0., 0.)     # What is the location of the top left corner. This is necessary to define the absolute location of the source and receivers
	nbpml= 40	      # "Infinite" domain by Absorbing Boundary Conditions (ABC) 

	vp_top = 1.5            # velocity in the top layer 
	vp_bottom = 2.5         # velocity in the bottom layer 

	t0 = 0.                 # Simulation starts a t=0
	#tn = 450.              # Simulation last 1 second (1000 ms)

	f0 = 0.010              # Source peak frequency is 10Hz (0.010 kHz)

	nreceivers = 101        # Number of receiver locations per shot 
	depth_source = 20.	# Source depth is 20m 
	depth_receiver = 20.	# Receiver depth is 20m 

	space_order = 4 
	time_order = 2

	##########################################################################################################################################

	# Define a velocity profile. The velocity is in km/s
	v = np.empty(shape, dtype=np.float32)
	v[:, :51] = vp_top
	v[:, 51:] = vp_bottom

	# Seismic model that encapsulates the properties. We also define the size of the absorbing layer
	model = Model(vp=v, origin=origin, shape=shape, spacing=spacing, space_order=space_order, nbpml=nbpml)

	# Time step from model grid spacing
	dt = model.critical_dt
	time_range = TimeAxis(start=t0, stop=tn, step=dt)

	# Define acquisition geometry: source
	src = RickerSource(name='src', grid=model.grid, f0=f0, npoint=1, time_range=time_range)
	# First, position source centrally in all dimensions, then set depth
	src.coordinates.data[0, :] = np.array(model.domain_size) * .5
	src.coordinates.data[0, -1] = depth_source

	# Create symbol for receivers
	rec = Receiver(name='rec', grid=model.grid, npoint=nreceivers, time_range=time_range)
	# Prescribe even spacing for receivers along the x-axis
	rec.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=nreceivers)
	rec.coordinates.data[:, 1] = depth_receiver

	# Define the wavefield u with the size of the model and the time dimension
	# corresponding to time-space-varying field (u, TimeFunction) and space-varying field (m, Function)
	u = TimeFunction(name="u", grid=model.grid, time_order=time_order, space_order=space_order)

	# Derive stencil (time marching updating equation) from symbolic equation
	# u(t+dt) is represented by u.forward
	pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
	stencil = Eq(u.forward, solve(pde, u.forward))

	# Source injection and receiver interpolation
	src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)
	rec_term = rec.interpolate(expr=u.forward)

	op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)
	op(time=time_range.num-1, dt=model.critical_dt)

	elapsed_time = time() - start_time
	print("Elapsed time: %.10f seconds." % elapsed_time)

	# Plot snapshots

	im = plt.imshow(np.transpose(u.data[336,40:-40,40:-40]), animated=False, vmin=-1e0, vmax=1e0,cmap=cm.jet, aspect=1, extent=[model.origin[0], model.origin[0] + 1e-3 * model.shape[0] * model.spacing[0], model.origin[1] + 1e-3*model.shape[1] * model.spacing[1], model.origin[1]])
	plt.xlabel('X position (km)',  fontsize=20)
	plt.ylabel('Depth (km)',  fontsize=20)
	plt.tick_params(labelsize=20)
	ims.append([im])

	plt.savefig("devito_acoustic_wave/snap%s.png"%tn)
	#plt.show()
	ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
                                repeat_delay=1000)
ani.save('devito_acoustic_wave/wave_propagation.mp4')
	###################################################################

	
