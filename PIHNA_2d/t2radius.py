import numpy as np

def t2radius(c,h,n,v,K,dx):
    T=(c+h+n+v)/K

    area1=np.count_nonzero(T>=0.8)*dx*dx
    rad_1=np.sqrt(area1/np.pi)

    area2=np.count_nonzero(T>=0.16)*dx*dx
    rad_2=np.sqrt(area2/np.pi)

    return rad_1,rad_2