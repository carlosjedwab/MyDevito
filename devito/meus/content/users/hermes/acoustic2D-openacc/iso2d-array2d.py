from mpl_toolkits.mplot3d import Axes3D    ##New Library required for projected 3d plots

import numpy
import time
from matplotlib import pyplot, cm
#%matplotlib inline

# vamos criar o grid 2D 
nx = 81
ny = 81
nt = 500    # numero de timesteps
c = 1       # velocidade da onda (igual em todo o dominio)
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx    # condição de estabilidade: dt < dx / 5

#cria eixos do grafico
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)

# por ser uma eq de segunda ordem no tempo, precisaremos de 3 copias da matriz
u_next = numpy.ones((ny, nx)) ##create a nx x ny for next value
u_val  = numpy.ones((ny, nx)) ##                ... for the actuar value
u_prev = numpy.ones((ny, nx)) ##                ... for the previous value

### Faz as condicoes iniciais 

#u_val[40,40]=2  # um ponto de pressão = 2 no centro do dominio

##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
##u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2 

##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
u_next[int(.5 / dy):int((.5+dy) / dy + 1),int(.5 / dx):int((.5+dx) / dx + 1)] = 2 

###Plot Initial Condition
##the figsize parameter can be used to produce different sized images
fig = pyplot.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')                      
X, Y = numpy.meshgrid(x, y)                            
#surf = ax.plot_surface(X, Y, u_next[:], cmap=cm.viridis)

t1 = time.time()
# loop no tempo 
for n in range(nt):
    u_prev = u_val.copy()
    u_val = u_next.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            value = 0.0
            #value += prev[gid + 1] - 2.0 * prev[gid] + prev[gid - 1];
            value += (u_val[i+1,j] - 2 * u_val[i,j] + u_val[i-1,j])/(dx * dx)
            #value += prev[gid + nCols] - 2.0 * prev[gid] + prev[gid - nCols];
            value += (u_val[i,j+1] - 2 * u_val[i,j] + u_val[i,j-1])/(dy * dy)
            #value *= dtDIVdxy * vel[gid];
            #value *= dt * dt * c * c
            value *= dt * dt * c * c
            # next[gid] = 2.0f * prev[gid] - next[gid] + value;
            u_next[i,j] = 2 * u_val[i,j] - u_prev[i,j] + value
#    if (n % 25)==0:
#        fig = pyplot.figure(figsize=(11, 7), dpi=100)
#        ax = fig.gca(projection='3d')
#        X, Y = numpy.meshgrid(x, y)
#        print(' *************   n = '+str(n))
#        surf = ax.plot_surface(X, Y, u_next[:], cmap=cm.viridis)

            
print('elapsed time (3 arrays + nested loops) '+str(time.time()-t1))

###########################   Com 2 veotres
# inicializa novamente

u_next = numpy.ones((ny, nx)) ##create a nx x ny for next value
u_prev = numpy.ones((ny, nx)) 
u_prev[int(.5 / dy):int((.5+dy) / dy + 1),int(.5 / dx):int((.5+dx) / dx + 1)] = 2 

#fig = pyplot.figure(figsize=(11, 7), dpi=100)
#ax = fig.gca(projection='3d')                      
#X, Y = numpy.meshgrid(x, y)                            
#surf = ax.plot_surface(X, Y, u_prev[:], cmap=cm.viridis)

t1 = time.time()
# loop no tempo 
for n in range(nt):
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            value = 0.0
            value += (u_prev[i+1,j] - 2 * u_prev[i,j] + u_prev[i-1,j])/(dx * dx)
            value += (u_prev[i,j+1] - 2 * u_prev[i,j] + u_prev[i,j-1])/(dy * dy)
            value *= dt * dt * c * c
            u_next[i,j] = 2 * u_prev[i,j] - u_next[i,j] + value
    swap = u_next
    u_next = u_prev
    u_prev = swap
#    if (n % 25)==0:
#        fig = pyplot.figure(figsize=(11, 7), dpi=100)
#        ax = fig.gca(projection='3d')
#        X, Y = numpy.meshgrid(x, y)
#        print(' *************   n = '+str(n))
#        surf = ax.plot_surface(X, Y, u_prev[:], cmap=cm.viridis)
        
print('nested loops + 2 swapped arrays - elapsed time is '+str(time.time()-t1))


#####################  Array numpy

# inicializa novamente

u_next = numpy.ones((ny, nx)) ##create a nx x ny for next value
u_prev = numpy.ones((ny, nx))
u_prev[int(.5 / dy):int((.5+dy) / dy + 1),int(.5 / dx):int((.5+dx) / dx + 1)] = 2

#fig = pyplot.figure(figsize=(11, 7), dpi=100)
#ax = fig.gca(projection='3d')
#X, Y = numpy.meshgrid(x, y)
#surf = ax.plot_surface(X, Y, u_prev[:], cmap=cm.viridis)

t1 = time.time()
# loop no tempo
for n in range(nt):
    u_next[1:,1:] = 2 * u_prev[1:,1:] - u_next[1:,1:] + (dt * dt * c * c * (
        ((u_prev[:+1,1:] - 2 * u_prev[1:,1:] + u_prev[:-1,1:])/(dx * dx))+
        ((u_prev[1:,:+1] - 2 * u_prev[1:,1:] + u_prev[1:,:-1])/(dy * dy)))
        )
    swap = u_next
    u_next = u_prev
    u_prev = swap
#    if (n % 25)==0:
#        fig = pyplot.figure(figsize=(11, 7), dpi=100)
#        ax = fig.gca(projection='3d')
#        X, Y = numpy.meshgrid(x, y)
#        print(' **   n = '+str(n))
#        surf = ax.plot_surface(X, Y, u_prev[:], cmap=cm.viridis)

print('Numpy arrays - elapsed time is '+str(time.time()-t1))

#fig = pyplot.figure(figsize=(11, 7), dpi=100)
#ax = fig.gca(projection='3d')
#X, Y = numpy.meshgrid(x, y)
#surf = ax.plot_surface(X, Y, u_prev[:], cmap=cm.viridis)


