#unset PETSC_DIR
#unset PETSC_ARCH
#vi ~/.bashcr nano ~/.bashcr
# import pdb; pdb.set_trace() #debug
#from fenics import * #Wildcard
#fn.UserExpression.__init__(self, pml, *args, **kwargs)
# UserExpression.super(self, *args, **kwargs)
#c.vector().get_local() => numpy.array
#import matplotlib.pyplot as plt
#DOLFIN_EPS = 1**(-10)
#print("Solving for time: ", t)

#serial
#real	11m58,897s
#user	15m50,985s
#sys	10m26,189s

#4cores
#real	20m51,278s
#user	80m39,158s
#sys	71m6,569s


# Global definitions
import math as mt
import fenics as fn
import numpy as np

########## Data Input ##########
# Folder name for results
Folder = "fenics_wave_acoustic"
FileRec = "Recept"
# Size mesh
refino = 0.4 #Refino max 2
nx = int(100*refino)
ny = int(100*refino)
# Domain geometry in km/s
Lx = 1.
Ly = 1.
# Propagation speed of the materials
velmat = np.array([1.5, 2.5])
# Size of the absorbing layer in km
pml = 0.1
# Number of timesteps
steps = 297
# Source frequency in Hz
f0 = 10.
# Sources definition
possou = np.array([[0.5], [0.98]])
# Receivers definition
posrec = np.array([[0.2, 0.4, 0.6, 0.8], [0.98, 0.98, 0.98, 0.98]])
########## End Data Input ##########

# Element number in absorbing layer
npmlx = int(np.max([1, mt.floor(nx*pml/Lx)])) # Element number in x
npmly = int(np.max([1, mt.floor(ny*pml/Ly)])) # Element number in y
print("Absorbing Layer - nelex:", npmlx, "and neley:", npmly)
# Create mesh
nelx = nx + 2*npmlx
nely = ny + 2*npmly
mesh = fn.RectangleMesh(fn.Point(0.0, 0.0), fn.Point(Lx + 2*pml, Ly  + 2*pml), nelx, nely)
# Normal vector
n = fn.FacetNormal(mesh)
# Minimum lenght of element
lmin = np.min([Lx/nx, Ly/ny])

# Solver parameters
forward_max_it = 10
forward_tol = 1e-12#fn.DOLFIN_EPS
relativ_tol = 1e-12#fn.DOLFIN_EPS

# Propagation speed
class vel_c(fn.UserExpression):
    """
    Mapping of propagation speed for domain with two different materials

    *Input arguments
        ** Globals:
        pml: Size of the absorbing layer in km
        velmat: Vector with propagation speeds in km/s

    *Class attributes:
        value: Field of propagation speed in domain
    """
    def __init__(self, *args, **kwargs):
        fn.UserExpression.__init__(self, *args, **kwargs)

    def eval(self, value, X):
        if X[1] > 0.5 + pml:
            value[:] = velmat[0]
        else:
            value[:] = velmat[1]

    def value_shape(self):
        return ()

# Damping properties
#(https://github.com/opesci/devito/blob/master/examples/seismic/model.py line codes 351-388)
class damping(fn.UserExpression):
    """
    Mapping of damping for absorbing layer

    *Input arguments
        ** Globals:
            pml: Size of the absorbing layer in km
            Lx, Ly: Domain dimensions
        ** Locals
            cmin: Minimum propagation speed in km/s

    *Class attributes:
        cmin: Minimum propagation speed in km/s
        valmax: Maximum damping
        value: Field of damping in domain

    *Class methods
        dist: Distant mapping in absorbing layer
        *Input arguments
            distx: Relative distance in x to physical domain boundary
            disty: Relative distance in y to physical domain boundary
        *Output arguments
            ref: Reference distance for evaluating of the damping
    """
    def __init__(self, cmin, *args, **kwargs):
        fn.UserExpression.__init__(self, *args, **kwargs)
        self.valmax = cmin * np.log(1.0 / 0.001) / (40.) #Maximum damping

    @staticmethod
    def dist(distx, disty):
        ref = np.linalg.norm(np.array([distx, disty]))
        return ref

    def eval(self, value, X):
        if X[0] <= pml or X[0] >= Lx + pml or X[1] <= pml or X[1] >= Ly + pml:

            if (X[0] <= pml or X[0] >= Lx + pml) and (X[1] <= pml or X[1] >= Ly + pml):

                if X[0] <= pml and X[1] <= pml:
                    ref = self.dist(X[0] - pml, X[1] - pml)
                elif X[0] >= Lx + pml and X[1] <= pml:
                    ref = self.dist(X[0] - (Lx + pml), X[1] - pml)
                elif X[0] <= pml and X[1] >= Ly + pml:
                    ref = self.dist(X[0] - pml, X[1] - (Ly + pml))
                elif X[0] >= Lx + pml and X[1] >= Ly + pml:
                    ref = self.dist(X[0] - (Lx + pml), X[1] - (Ly + pml))

            elif X[0] <= pml:
                ref = self.dist(pml - X[0], 0)
            elif X[0] >= Lx + pml:
                ref = self.dist(X[0] - (Lx + pml), 0)
            elif X[1] <= pml:
                ref = self.dist(0, pml - X[1])
            elif X[1] >= Ly + pml:
                ref = self.dist(0, X[1] - (Ly + pml))

            value[:] = self.valmax*(ref/pml - np.sin(2*np.pi*ref)/(2*np.pi))
        else:
            value[:] = 0

    def value_shape(self):
        return ()

#dt computation
def critical_dt(minL, cmax, steps):
    """
    dt computation with CFL condtion given by dt <= coeff * h / (max(c))

    *Input arguments
        minl: Minimum lenght of element
        cmax: Maximum propagation speed in km/s
        steps: Number of timesteps
    *Output arguments
        dt: Timestep size
    """
    coeff = 0.42
    dt = coeff * minL / cmax
    return (1./steps) * np.max([int(steps * dt), 1])

# Excitation
class RickerSource(fn.UserExpression):
    """
    Ricker source generation for modeling excitation

    *Input arguments
        ** Globals:
        f0: Source frequency in kHz
        possou: Sources localization
        pml: Size of the absorbing layer in km
        ** Locals
        t: Current time
        tol: Tolerance for location in the mesh

    *Class attributes:
        t: Current time
        tol: Tolerance for location in the mesh
        value: Excitation at point in domain
    """
    def __init__(self, t, tol, *args, **kwargs):
        fn.UserExpression.__init__(self, *args, **kwargs)
        self.t = t
        self.tol = tol

    def eval(self, value, X):
        r = (np.pi * f0 * (self.t - 1./f0))**2
        amp = (1. - 2.*r)*np.exp(-r)
        value[:] = 0

        for i in range(np.shape(possou)[1]):
            xsou = possou[0, i] + pml
            ysou = possou[1, i] + pml
            if fn.near(X[0], xsou, self.tol) and fn.near(X[1], ysou, self.tol):
                value[:] = amp

    def value_shape(self):
        return ()

# #Receiver evaluation
# def Receiver(t, posrec, pml, w):
    # """
    # Data aquisition of the receivers

    # *Input arguments
        # t: Current time
        # posrec: Receivers localization
        # pml: Size of the absorbing layer in km
        # w: Variable of state
    # *Output arguments
        # datarec: Receiver history no current time t
    # """
    # print("Recording at Receivers")
    # datarec = ["{:.5e}".format(t)]
    # for i in range(posrec.shape[1]):
        # try:
            # datarec.append("{:=13.5e}".format(w(fn.Point(posrec[0, i] + pml, posrec[1, i] + pml))))
        # except:
            # #print(fn.MPI.rank(fn.MPI.comm_world))
            # pass

    # datarec = str(datarec)[2:-2].replace("'", "").replace(",", " ") + "\n"
    # return datarec

# # Output Receivers files
# def WriteToDisk(histrec, fileName):
    # """
    # File generation for receivers data

    # *Input arguments
        # histrec: Final history of the receivers
        # fileName: File name for results of the receivers
    # """
    # mpi_number = fn.MPI.rank(fn.MPI.comm_world)
    # with open(fileName + "_mpi" + str(mpi_number) +".txt", "w") as filerec:
        # filerec.write(histrec)

# Forward problem
def forward(lmin, pml, steps, Folder):
    """
    Algorithm to solve the forward problem

    *Input arguments
        minl: Minimum lenght of element
        pml: Size of the absorbing layer in km
        steps: Number of timesteps
        Folder: Folder name for results
    *Output arguments
        c: Field of propagation speed in domain
        eta: Field of damping in domain
        histrec: Historical data receivers
    """
    histrec = ''
    # Create and define function space
    V = fn.FunctionSpace(mesh, "CG", 1)
    w = fn.TrialFunction(V)
    v = fn.TestFunction(V)

    # Boundary conditions u0 = 0.0
    bc = fn.DirichletBC(V, 0.0, fn.DomainBoundary())

	# State variable and source space functions
    w = fn.Function(V, name="Veloc")
    w_ant1 = fn.Function(V)
    w_ant2 = fn.Function(V)
    q = fn.Function(V, name="Sourc")

    # Velocity mapping
    c_func = vel_c()
    c = fn.interpolate(c_func, V)
    velp_file = fn.File(Folder + "/Velp.pvd")
    velp_file << c

	# Damping mapping
    damping_func = damping(c.vector().get_local().min())
    eta = fn.interpolate(damping_func, V)
    damp_file = fn.File(Folder + "/Damp.pvd")
    damp_file << eta

    # Current time we are solving for
    t = 0.
    # Number of timesteps solved
    ntimestep = int(0)
	# dt computation
    dt = critical_dt(lmin, c.vector().get_local().max(), steps)
	# Final time
    T = steps*dt
    print("Time step:", dt, "s")

    # "Bilinear term"
    def aterm(u, v, dt, c, eta):
        termgrad = dt**2*fn.inner(c, c)*fn.inner(fn.grad(u), fn.grad(v))*fn.dx
        termvare = (1.0 + dt*eta*fn.inner(c, c))*fn.inner(u, v)*fn.dx
        termboun = (dt**2*c*c)*fn.inner(fn.grad(u), n)*v*fn.ds
        return  termgrad + termvare - termboun

    # Source term
    def lsourc(u_ant1, u_ant2, v, dt, c, q, eta):
        termua1 = (-2. + dt*eta*fn.inner(c, c))*fn.inner(u_ant1, v)*fn.dx
        termua2 = fn.inner(u_ant2, v)*fn.dx
        termsou = (dt**2*fn.inner(c, c))*fn.inner(q, v)*fn.dx
        return  termua1 + termua2 - termsou

	# Output file names
    Exct = fn.File(Folder+"/Exct.pvd")
    Wave = fn.File(Folder+"/Wave.pvd")

    while True:
        # Update the time it is solving for
        print("Step: {0:1d} - Time: {1:1.4f}".format(ntimestep, t), "s")

        # Ricker source for time t
        ExcSource = RickerSource(t, tol=lmin/2)
        q.assign(fn.interpolate(ExcSource, V))

        # Time integration loop
        if ntimestep < 1:
            g = fn.interpolate(fn.Expression('0.0', degree=2), V)
            w.assign(g) # assign initial guess for solver
            w_ant1.assign(g) # assign initial guess for solver
            w_ant2.assign(g) # assign initial guess for solver
        else:
            a = aterm(w, v, dt, c, eta)
            L = lsourc(w_ant1, w_ant2, v, dt, c, q, eta)
            #solve equation
            fn.solve(a + L == 0, w, bcs=bc, solver_parameters={"newton_solver":
                {"maximum_iterations": forward_max_it, "absolute_tolerance": forward_tol,
                 "relative_tolerance": relativ_tol}})

        # Cycling the variables
        w_ant2.assign(w_ant1)
        w_ant1.assign(w)

        #Output files
        # histrec += Receiver(t, posrec, pml, w)
        Exct << (q, t)
        Wave << (w, t)

        t += dt
        ntimestep += 1
        
        if t >= T:
            break
    
    return c, eta, histrec

#Solving the forward problem
c, eta, histrec = forward(lmin, pml, steps, Folder)

# # Output Receivers files
# WriteToDisk(histrec, Folder + "/" + FileRec)
