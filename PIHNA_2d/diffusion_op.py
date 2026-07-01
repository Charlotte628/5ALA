import numpy as np
from numba import jit

@jit
def diffusion_op(u,D,dx,Weight1,Weight2,Origin1,Origin2,Terminus1,Terminus2):

    # flux across horizonal faces (x-direction)
    flux1=D*Weight1*(u[Origin1]-u[Terminus1])/(dx**2)
    #flux across vertical faces (y-direction)
    flux2=D*Weight2*(u[Origin2]-u[Terminus2])/(dx**2)

    b=np.zeros_like(u)
    for i in range(len(Origin1)):
        b[Origin1[i]]-=flux1[i]
    for i in range(len(Terminus1)):
        b[Terminus1[i]]+=flux1[i]
    for i in range(len(Origin2)):
        b[Origin2[i]]-=flux2[i]
    for i in range(len(Terminus2)):
        b[Terminus2[i]]+=flux2[i]
    return b