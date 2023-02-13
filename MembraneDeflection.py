from mshr import *
from fenics import *
import matplotlib.pyplot as plt

from vedo.dolfin import plot


#mesh over unit disk created with mshr tool
domain = Circle(Point(0,0), 1)
mesh= generate_mesh(domain, 64)
V= FunctionSpace(mesh, 'Lagrange', 1)

#load

beta= 8
R0= 0.6
u_D = Constant(0.0)

def on_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, on_boundary)

p = Expression('4*exp(-pow(beta,2)*(pow(x[0], 2) +pow(x[1]-R0, 2)))', degree=1, beta=beta, R0=R0)

w=TrialFunction(V)
v= TestFunction(V)

a= dot(grad(w), grad(v))*dx

L=p*v*dx

w= Function(V)

solve(a==L, w, bc)
p = interpolate(p, V)
plt.figure()
plt.subplot(1,2,1)

plot(w, title= 'Deflection')

plt.subplot(1,2,2)
plot(p, title= 'Load')

plt.show()