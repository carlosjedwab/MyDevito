from examples.seismic import Model, demo_model, AcquisitionGeometry, plot_velocity, plot_perturbation
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.model import SeismicModel
import numpy as np
import copy
import os

def downscale_vp(vp, spacing, scale):
    # Method to scale down a given matrix's shape, respectively lowering it's resolution
    
    high_res_shape = vp.shape
    high_res_spacing = spacing
    
    low_res_shape = (int(high_res_shape[0]/scale), int(high_res_shape[1]/scale))
    low_res_spacing = (high_res_spacing[0]*scale, high_res_spacing[1]*scale)
    low_res_vp = np.zeros(low_res_shape)
    for i in range(0, low_res_vp.shape[0]):
        for j in range(0, low_res_vp.shape[1]):
            low_res_vp[i][j] = vp[int(i*scale)][int(j*scale)]
    return low_res_vp, low_res_spacing

def smoothen_model(model, sigma = 6.0):
    # Method to return a smoothen version of a given model
    
    s_model = copy.deepcopy(model)
    if sigma != 1:
        s_model = copy.deepcopy(model)
        s_model.smooth(model.physical_params(), sigma)
    return s_model

def low_res_marmousi(scale = 1.0):
    # Custom method for demo_model('marmousi2d-isotropic'...) from examples.seismic.model
    
    high_res_shape = (1601, 401)
    high_res_spacing = (7.5, 7.5)
    origin = (0., 0.)
    nbl = 20

    # Read 2D Marmousi model from devitocodes/data repo
    data_path = "data/"
    if data_path is None:
        raise ValueError("Path to devitocodes/data not found! Please specify with "
                         "'data_path=<path/to/devitocodes/data>'")
    path = os.path.join(data_path, 'Simple2D/vp_marmousi_bi')
    v = np.fromfile(path, dtype='float32', sep="")
    v = v.reshape(high_res_shape)
    
    # Cut the model to make it slightly cheaper
    v = v[301:-300, :]
    
    # Rescale the model to make it cheaper
    v, low_res_spacing = downscale_vp(v, high_res_spacing, scale)

    return SeismicModel(space_order=2, vp=v, origin=origin, shape=v.shape,
                        dtype=np.float32, spacing=low_res_spacing, nbl=20,
                        bcs="damp", fs=False)

def demo_fwi(preset, **kwargs):
    # Method to return preset-setups for a fwi aplication
    
    shape = kwargs.pop('shape', (101, 101))                       # Number of grid point (nx, nz)
    spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))  # Grid spacing in m
    origin = kwargs.pop('origin', tuple([0. for _ in shape]))     # Defines relative source and receiver locations   
    nshots = kwargs.pop('nshots', int(shape[0]/2))                # One every two grid points
    nreceivers = kwargs.pop('nreceivers', int(shape[0]))          # One recevier every grid point
    t0 = kwargs.pop('t0', 0.)                                     # Simulation starts at t=0
    tn = kwargs.pop('tn', 3500.)                                  # Simulation last 3.5 seconds (3500 ms)
    f0 = kwargs.pop('f0', 0.025)                                  # Source peak frequency is 25Hz (0.025 kHz)
    
    if preset.lower() in ['marmousi2d-isotropic', 'm2d']:
        shape = (1601, 401)
        spacing = (7.5, 7.5)
        origin = (0., 0.)
        nshots = 301
        nreceivers = 601
        nbl = kwargs.pop('nbl', 20)
        resolution_scale = kwargs.pop('resolution_scale', 3.0)    # Scale to which the shape is rescaled
        filter_sigma = kwargs.pop('filter_sigma', 2.0)            # Sigma to which the data is smoothened  
        
        # Build the model based on the preset data
        if resolution_scale != 1:
            true_model = low_res_marmousi(resolution_scale)
        else:
            true_model = demo_model('marmousi2d-isotropic', data_path='data/', grid=None, nbpml=20)
        
        # Create initial model by smooth the boundaries
        fwi_model0 = smoothen_model(true_model, filter_sigma)
        
        # Position source
        src_coordinates = np.empty((1, 2))
        src_coordinates[0, :] = np.array(true_model.domain_size) * .5
        src_coordinates[0, 1] = 20.  # Depth is 20m
        
        # Initialize receivers for synthetic and imaging data
        rec_coordinates = np.empty((nreceivers, 2))
        rec_coordinates[:, 0] = np.linspace(0, true_model.domain_size[0], num=nreceivers)
        rec_coordinates[:, 1] = 20. # Depth(m)
        
        # Prepare the varying source locations sources
        source_locations = np.empty((nshots, 2), dtype=np.float32)
        source_locations[:, 1] = 30.
        source_locations[:, 0] = np.linspace(0., 7500, num=nshots)
        
        # Ready up the Geometry
        geometry = AcquisitionGeometry(true_model, rec_coordinates, src_coordinates,
                                       t0, tn, f0=f0, src_type='Ricker')
        
        # Construct the Solver
        solver = AcousticWaveSolver(true_model, geometry, space_order=4)
        
        # Attribute the number of fwi iterations
        fwi_iterations = 20
        
    elif preset.lower() in ['circle-isotropic', 'c2d']:
        nshots = 9
        tn = 1000.
        f0 = 0.010
        
        # Build the model based on the preset data
        true_model = demo_model('circle-isotropic', vp_circle=3.0, vp_background=2.5,
                         origin=origin, shape=shape, spacing=spacing, nbl=40)
        
        # Create initial model by smooth the boundaries
        fwi_model0 = demo_model('circle-isotropic', vp_circle=2.5, vp_background=2.5,
                         origin=origin, shape=shape, spacing=spacing, nbl=40,
                         grid = true_model.grid)
        
        # Position source
        src_coordinates = np.empty((1, 2))
        src_coordinates[0, :] = np.array(true_model.domain_size) * .5
        src_coordinates[0, 0] = 20.  # Depth is 20m
        
        # Initialize receivers for synthetic and imaging data
        rec_coordinates = np.empty((nreceivers, 2))
        rec_coordinates[:, 1] = np.linspace(0, true_model.domain_size[0], num=nreceivers)
        rec_coordinates[:, 0] = 980.
        
        # Prepare the varying source locations sources
        source_locations = np.empty((nshots, 2), dtype=np.float32)
        source_locations[:, 0] = 30.
        source_locations[:, 1] = np.linspace(0., 1000, num=nshots)
        
        # Ready up the Geometry
        geometry = AcquisitionGeometry(true_model, rec_coordinates, src_coordinates,
                                       t0, tn, f0=f0, src_type='Ricker')
        
        # Construct the Solver
        solver = AcousticWaveSolver(true_model, geometry, space_order=4)
        
        # Attribute the number of fwi iterations
        fwi_iterations = 5
        
    
    # Show the plots
    if kwargs.pop('show_plots', False):
        print("True Model:")
        plot_velocity(true_model)
        print("FWI Model 0:")
        plot_velocity(fwi_model0)
        #print("True Model ad FWI Model 0 difference:")
        #plot_perturbation(fwi_model0, true_model)
        print("Sources and receivers positions:")
        plot_velocity(true_model, source=geometry.src_positions, receiver=geometry.rec_positions[::4, :])
        print("Sources locations:")
        plot_velocity(true_model, source=source_locations)
        print("Geometry:")
        geometry.src.show()
        
    return true_model, fwi_model0, nshots, nreceivers, src_coordinates, rec_coordinates,\
           source_locations, geometry, solver, fwi_iterations


'''
Usage example:
>>> model, model0, nshots, nreceivers, src_coordinates, rec_coordinates, source_locations,\
geometry, solver, fwi_iterations = demo_fwi('c2d', show_plots=True)
'''