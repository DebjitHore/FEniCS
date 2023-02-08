from fenics import *
import fenics as fe
import matplotlib.pyplot as plt
from vedo.dolfin import plot



#Mesh and Function Space

mesh= UnitSquareMesh(8,8)
V= FunctionSpace(mesh, 'Lagrange', 1)

# Boundary Condition

u_D = Expression('1+x[0]*x[0]+2*x[1]*x[1]', degree=2)

def on_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, on_boundary)

#Variational Problem

u=TrialFunction(V)
v= TestFunction(V)
f= Constant(-6)
a= dot(grad(u), grad(v))*dx
L= f*v*dx

#Compute Solution

u= Function(V)
solve(a==L, u, bc)


#plot solution and mesh
plt.plot()
plot(u, mode='color', vmin=1, vmax=4, style =1)

plt.show()


error_L2= errornorm(u_D, u, 'L2')

vertex_values_uD= u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)

import numpy as np
error_max = np.max(np.abs(vertex_values_uD-vertex_values_u))

print('error_L2= ', error_L2)
print('error_max=', error_max)

