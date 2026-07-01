import numpy as np
from numba import jit

@jit
def reaction(c,h,n,v,a,D_c,D_h_mul,rho,beta_mul,gamma,alpha_h_mul,alpha_n,D_v,D_a,delta_c,delta_h,mu,q,lam,K,Km,P_cw,P_hw,P_cg,P_hg,v0,u0):
    D_h=D_h_mul*D_c
    beta=beta_mul*rho
    alpha_h=alpha_h_mul*beta
    omega=lam/v0
    P_c=P_cw
    P_h=P_hw

    # calculating reaction terms
    T=(c+h+n+v)/K
    #V=np.zeros_like(T)
    #print(f'reaction\n{c}')
    #for j in range(v.size):
        #if v[j]==0:
            #V[j]=0
        #else:
            #V[j]=v[j]/(v[j]+((P_c*c[j]+P_h*h[j])/((u0/v0)-1)))
    if v==0:
        V=0
    else:
        V=v/(v+(P_c*c+P_h*h)/((u0/v0)-1))

    dc=rho*c*(1-T)+gamma*h*V-beta*c*(1-V)-alpha_n*n*c/K
    #print(f'{dc}')
    dh=beta*c*(1-V)-gamma*h*V-(alpha_h*h*(1-V)+alpha_n*n*h/K)
    dn=alpha_h*h*(1-V)+alpha_n*n*(c+h+v)/K
    dv=mu*(a/(Km+a))*v*(1-T)-alpha_n*n*v/K
    da=delta_c*c+delta_h*h-q*mu*(a/(Km+a))*v*(1-T)-omega*a*v-lam*a

    return dc,dh,dn,dv,da