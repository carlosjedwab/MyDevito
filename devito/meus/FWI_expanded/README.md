# File tree

```
FWI_expanded
|
├───model_datas
|	├───datas
|	|	└───...
|	├───utils.py: resize, smooth
|	└───segyio_extended.py: read_segy, write_segy, duplicate_segy
|
├───fwi_expanded
|	├───solvers
|	|	├───specific_solver_builder.py: build_solver
|	|	├───class_devito_solver.py: class DevitoSolver(AcousticWaveSolver)
|	|	├───class_custom_solver.py: class CustomSolver(AcousticWaveSolver)
|	|	└───...
|	└───fwi_methods
|		├───fwi_params_presets.py: fwi_demo_params
|		└───fwi_core.py: compute_residual, fwi_gradient, update_with_box, apply_fwi, plot_objective
|
└───aplications
	├───example_01.ipynb
	└───...
```
		
	
		
	
