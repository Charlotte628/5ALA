import numpy as np
from scipy.sparse.linalg import cg
from typing import Dict,Any

from PIHNA_2d.diffusion_op import diffusion_op
from PIHNA_2d.reaction import reaction
from PIHNA_2d.t2radius import t2radius

from scipy.sparse.linalg import LinearOperator

from numba import jit

@jit
def numerical_jacobian_jit(Q,D_c,D_h_mul,rho,beta_mul,gamma,alpha_h_mul,alpha_n,D_v,D_a,delta_c,delta_h,mu,q,lam,K,Km,P_cw,P_hw,P_cg,P_hg,v0,u0,eps=1e-6):
    J=np.zeros((5,5))
    c=Q[0]
    h=Q[1]
    n=Q[2]
    v=Q[3]
    a=Q[4]
    T=(Q[0]+Q[1]+Q[2]+Q[3])/K #c+h+n+v
    beta=beta_mul*rho
    alpha_h=alpha_h_mul*beta
    omega=lam/v0
    P_h=P_hw
    P_c=P_cw
    if v==0:
        V=0
        dVdc=0
        dVdh=0
    elif c+h==0:
        dVdv=0
    else:
        V=v/(v+((P_c*c+P_h*h)/((u0/v0)-1)))
        dVdc=-P_c*v/(((u0/v0)-1)*(v+(P_c*c+P_h*h)/(u0/v0 -1))**2)
        dVdh=-P_h*v/(((u0/v0)-1)*(v+(P_c*c+P_h*h)/(u0/v0 -1))**2)
        dVdv=(P_c*c+P_h*h)/(((u0/v0)-1)*(v+(P_c*c+P_h*h)/(u0/v0 -1))**2)
    J[0,0]=rho*(1-T)-rho*c/K + gamma*h*dVdc-beta*(1-V)+beta*c*dVdc-alpha_n*c/K#rho*(1-(h+n+v)/K)-gamma*P_c*h*v/((u0/v0 -1)*(v+(P_c*c+P_h*h)/(u0/v0 -1))**2) - beta*(1-V)-P_c*beta*c*v/((u0/v0 -1)*(v+(P_c*c+P_h*h)/(u0/v0 -1))**2)-alpha_n*n/K#c,c
    #print('cc')
    J[0,1]=-rho*c/K +gamma*V+gamma*h*dVdh+beta*c*dVdh#gamma*V-rho*c/K + (beta-gamma*h)*P_h*v/((u0/v0 -1)*(v+(P_c*c+P_h*h)/(u0/v0-1))**2)#c,h
    #print('ch')
    J[0,2]=-(rho+alpha_n)*c/K#-c*(rho+alpha_n)/K #c,n
    #print('cn')
    J[0,3]=-rho*c/K + gamma*h*dVdv+beta*c*dVdv#(P_c*c+P_h*h)/((u0/v0 -1)*(v+(P_c*c+P_h*h)/(u0/v0-1))**2)*(gamma*h+beta*c)-rho*c/K #c,v
    #print('cv')
    J[0,4]=0 #c,a
    #print('ca')

    J[1,0]=-gamma*h*dVdc+beta*(1-V)-beta*c*dVdc+alpha_h*h*dVdc#(P_c*v/((u0/v0 -1)*(v+(P_c*c+P_h*h)/(u0/v0-1))**2))*(gamma*h+beta*c-alpha_h*h)+beta*(1-V) #h,c
    #print('hc')
    J[1,1]=-gamma*V-gamma*h*dVdh-beta*c*dVdh-alpha_h*(1-V)+alpha_h*h*dVdh-alpha_n*n/K#(P_h*v/((u0/v0 -1)*(v+(P_c*c+P_h*h)/(u0/v0-1))**2))*(gamma*h+beta*c-alpha_h*h)-gamma*V-alpha_h*(1-V)-alpha_n*n/K #h,h
    #print('hh')
    J[1,2]= -alpha_n*h/K#h,n
    #print('hn')
    J[1,3]=-gamma*h*dVdv-beta*c*dVdv+alpha_h*h*dVdv#(P_c*c+P_h*h)/((u0/v0-1)*(v+(P_c*c+P_h*h)/(u0/v0-1))**2)*(alpha_h*h-gamma*h-beta*c)#h,v
    #print('hv')
    J[1,4]=0#h,a
    #print('ha')

    J[2,0]=-alpha_h*h*dVdc+alpha_n*n/K#alpha_h*P_c*v*h/((u0/v0-1)*(v+(P_c*c+P_h*h)/(u0/v0-1))**2) + alpha_n*n/K #n,c
    #print('nc')
    J[2,1]=alpha_h*(1-V)-alpha_h*h*dVdh+alpha_n*n/K#alpha_h*(1-V)+alpha_h*P_h*v*h/((u0/v0-1)*(v+(P_c*c+P_h*h)/(u0/v0-1))**2)+alpha_n*n/K #n,h
    #print('nh')
    J[2,2]=alpha_n*(c+h+v)/K #n,n
    #print('nn')
    J[2,3]=-alpha_h*h*dVdv+alpha_n*n/K#alpha_n*n/K - alpha_h*h*(P_c*c+P_h*h)/((u0/v0-1)*(v+(P_c*c+P_h*h)/(u0/v0-1))**2)#n,v
    #print('nv')
    J[2,4]=0
    #print('nn')

    J[3,0]=-mu*v*a/(K*(Km+a)) #v,c
    #print('vc')
    J[3,1]=-mu*v*a/(K*(Km+a)) #v,h
    #print('vh')
    J[3,2]=-mu*v*a/(K*(Km+a)) - alpha_n*v/K #v,n
    #print('vn')
    J[3,3]=mu*a*(1-T)/(Km+a) - mu*a*v/(K*(Km+a)) - alpha_n*n/K #v,v
    #print('vv')
    J[3,4]=mu*(Km/(Km+a)**2)*v*(1-T)#Km*mu*v*(1-T)/((Km+a)**2) #v,a
    #print('va')

    J[4,0]=delta_c+q*mu*a*v/(K*(Km+a)) #a,c
    #print('ac')
    J[4,1]=delta_h+q*mu*a*v/(K*(Km+a)) #a,h
    #print('ah')
    J[4,2]=q*mu*v*a/(K*(Km+a)) #a,n
    #print('an')
    J[4,3]=q*mu*v*a/(K*(Km+a)) - q*mu*(1-T)*a/(Km+a) - omega*a
    #print('av')
    J[4,4]=-q*mu*v*(1-T)*Km/((Km+a)**2) - omega*v-lam
    #print('aa')
    '''
    J=np.zeros((5,5))
    f0=np.array(reaction(Q[0],Q[1],Q[2],Q[3],Q[4],D_c,D_h_mul,rho,beta_mul,gamma,alpha_h_mul,alpha_n,D_v,D_a,delta_c,delta_h,mu,q,lam,K,Km,P_cw,P_hw,P_cg,P_hg,v0,u0))
    for j in range(5):
        dq=np.zeros(5)
        dq[j]=eps
        f1=np.array(reaction(Q[0]+dq[0],Q[1]+dq[1],Q[2]+dq[2],Q[3]+dq[3],Q[4]+dq[4],D_c,D_h_mul,rho,beta_mul,gamma,alpha_h_mul,alpha_n,D_v,D_a,delta_c,delta_h,mu,q,lam,K,Km,P_cw,P_hw,P_cg,P_hg,v0,u0))
        J[:,j]=(f1-f0)/eps
    for i in range(5):
        for j in range(5):
            if np.isinf(J[i,j])==True:
                #print(f'Indices: {i,j}')
                #print(f'c: {c}')
                print(f'h: {h}')
                print(f'n: {n}')
                print(f'v: {v}')
                print(f'a: {a}')
                #print(f'{q*mu*v*a/(K*(Km+a))}')
                #print(q*mu*(1-T)*a/(Km+a))
                #print(1-T)
                #print(1/(Km+a))
                #print(omega*a)
    '''
    if True in np.isinf(J):
        print('Not finite jacob')
    return J


def numerics_2d(
        Dweight1,Dweight2,
        Origin1,Origin2,
        Terminus1,Terminus2,
        n1,n2,ind_nz,boundary,
        IC,dayspersave, t_finish,
        radiussave, radius_final, 
        P
):
    dt=float(P['dt'])
    dx=float(P['dx'])

    D_c=float(P['D_c'])
    rho=float(P['rho'])
    K=float(P['K'])
    Km=float(P['Km'])

    D_h_mul=float(P['D_h_mul'])
    D_h=D_h_mul*D_c
    beta_mul=float(P['beta_mul'])
    beta=beta_mul*rho
    gamma=float(P['gamma'])#/float(P['rho'])
    alpha_h_mul=float(P['alpha_h_mul'])
    lam=float(P['lam'])#/float(P['rho'])
    v0=float(P['v0'])
    u0=float(P['u0'])
    alpha_h=alpha_h_mul*beta
    omega=lam/v0
    alpha_n=float(P['alpha_n'])#/float(P['rho'])
    D_v=float(P['D_v'])#/float(P['D_c'])
    D_a=float(P['D_a'])#/float(P['D_c'])
    delta_c=float(P['delta_c'])#*float(P['K'])/(float(P['Km'])*float(P['rho']))
    delta_h=float(P['delta_h'])#*float(P['K'])/(float(P['Km'])*float(P['rho']))
    mu=float(P['mu'])#/float(P['rho'])
    q=float(P['q'])#*float(P['K'])/float(P['Km'])

    P_cw=float(P['P_cw'])
    P_hw=float(P['P_hw'])
    P_cg=float(P['P_cg'])
    P_hg=float(P['P_hg'])
    P_c=P_cw
    P_h=P_hw

    c=IC['c0'].ravel(order='F').copy().astype(float)#/float(P['K'])
    h=IC['h0'].ravel(order='F').copy().astype(float)#/float(P['K'])
    n=IC['n0'].ravel(order='F').copy().astype(float)#/float(P['K'])
    v=IC['v0'].ravel(order='F').copy().astype(float)#/float(P['K'])
    a=IC['a0'].ravel(order='F').copy().astype(float)#/float(P['K'])

    N=n1*n2

    saved_times=[]
    saved_c=[]
    saved_h=[]
    saved_n=[]
    saved_v=[]
    saved_a=[]
    saved_rad1=[]
    saved_rad2=[]

    def save_state(t):
        c_grid=c.reshape((n1,n2),order='F')
        h_grid=h.reshape((n1,n2),order='F')
        n_grid=n.reshape((n1,n2),order='F')
        v_grid=v.reshape((n1,n2),order='F')
        a_grid=a.reshape((n1,n2),order='F')
        rad1,rad2=t2radius(c_grid,h_grid,n_grid,v_grid,K,dx)

        saved_times.append(t)
        saved_c.append(c.copy())
        saved_h.append(h.copy())
        saved_n.append(n.copy())
        saved_v.append(v.copy())
        saved_a.append(a.copy())
        saved_rad1.append(rad1)
        saved_rad2.append(rad2)

    # save initial state
    t_dim=0
    t_step=0
    save_state(t_dim)

    if dayspersave!=0:
        steps_per_save=int(round(dayspersave)/dt)
    else: 
        next_radius_target=radiussave
    
    while True:
        if t_step%20==0:
            print(f'----------{t_dim}---------')
        t_step+=1
        t_dim=t_step*dt

        T=(c+h+n+v)/K
        V=np.zeros_like(T)
        for j in range(v.size):
            if v[j]==0:
                V[j]=0
            else:
                V[j]=v[j]/(v[j]+((P_c*c[j]+P_h*h[j])/((u0/v0)-1)))
            #V=v/(v+((P_c*c+P_h*h)/((u0/v0)-1)))

        DC=(1-T)
        DC1=0.5*(DC[Origin1]+DC[Terminus1])
        DC2=0.5*(DC[Origin2]+DC[Terminus2])
        #Diffusion
        ## Normoxic
        N=c.size
        A_c=LinearOperator(shape=(N,N),
                              matvec=lambda x:x-dt*diffusion_op(x,D_c,dx,DC1*Dweight1,DC2*Dweight2,Origin1,Origin2,Terminus1,Terminus2))
        b_c=dt*diffusion_op(c,D_c,dx,DC1*Dweight1,DC2*Dweight2,Origin1,Origin2,Terminus1,Terminus2)
        ## Hypoxic
        A_h=LinearOperator(shape=(N,N),
                              matvec=lambda x:x-dt*diffusion_op(x,D_h,dx,DC1*Dweight1,DC2*Dweight2,Origin1,Origin2,Terminus1,Terminus2))
        b_h=dt*diffusion_op(h,D_h,dx,DC1*Dweight1,DC2*Dweight2,Origin1,Origin2,Terminus1,Terminus2)
        ## Necrotic-doesn't diffuse
        ## Vasculature 
        A_v=LinearOperator(shape=(N,N),
                              matvec=lambda x:x-dt*diffusion_op(x,D_v,dx,DC1*Dweight1,DC2*Dweight2,Origin1,Origin2,Terminus1,Terminus2))
        b_v=dt*diffusion_op(v,D_v,dx,DC1*Dweight1,DC2*Dweight2,Origin1,Origin2,Terminus1,Terminus2)
        ## Angiogenic Factor
        A_a=LinearOperator(shape=(N,N),
                              matvec=lambda x:x-dt*diffusion_op(x,D_a,dx,Dweight1,Dweight2,Origin1,Origin2,Terminus1,Terminus2))
        b_a=dt*diffusion_op(a,D_a,dx,Dweight1,Dweight2,Origin1,Origin2,Terminus1,Terminus2)

        ##Conjugate Gradient
        dc,info=cg(A_c,b_c,rtol=1e-6,atol=0,maxiter=500)
        dh,info=cg(A_h,b_h,rtol=1e-6,atol=0,maxiter=500)
        dv,info=cg(A_v,b_v,rtol=1e-6,atol=0,maxiter=500)
        da,info=cg(A_a,b_a,rtol=1e-6,atol=0,maxiter=500)

        c=c+dc
        h=h+dh
        v=v+dv
        a=a+da

        c=np.maximum(c,0)
        h=np.maximum(h,0)
        v=np.maximum(v,0)
        a=np.maximum(a,0)

        #print('after diffusion: ')
        #print(np.max(c))
        #print(np.max(h))
        #print(np.max(n))
        #print(np.max(v))
        #print(np.max(a))

        #reaction step
        if c.max()>P['K']:
            print('c= ',c)
        if True in np.isnan(c):
            print('nan, c')
        if True in np.isnan(h):
            print('nan, h')
        if True in np.isnan(n):
            print('nan, n')
        if True in np.isnan(v):
            print('nan, v')
        if True in np.isnan(a):
            print('nan, a')
        if True in np.isinf(c):
            print('inf, c')
        if True in np.isinf(h):
            print('inf, h')
        if True in np.isinf(n):
            print('inf, n')
        if True in np.isinf(v):
            print('inf, v')
        if True in np.isinf(a):
            print('inf, a')
        c,h,n,v,a=reaction_step(c,h,n,v,a,dt,D_c,D_h_mul,rho,beta_mul,gamma,alpha_h_mul,alpha_n,D_v,D_a,delta_c,delta_h,mu,q,lam,K,Km,P_cw,P_hw,P_cg,P_hg,v0,u0)
        rad1,rad2=t2radius(
            c.reshape((n1,n2),order='F'),
            h.reshape((n1,n2),order='F'),
            n.reshape((n1,n2),order='F'),
            v.reshape((n1,n2),order='F'),
            K,dx)
        
        #Save
        if dayspersave!=0:
            if t_step%steps_per_save==0:
                save_state(t_dim)
            if t_dim>=t_finish-1e-12:
                break
        else:
            if rad2>=next_radius_target:
                save_state(t_dim)
                next_radius_target+=radiussave
            if rad2>=radius_final:
                break

        #Kill Condition
        if rad2>4.5:
            save_state(t_dim)
            break
    
    return {'time':saved_times,'c':np.vstack(saved_c),'h':np.vstack(saved_h),'n':np.vstack(saved_n),'v':np.vstack(saved_v),'a':np.vstack(saved_a),'rad1':saved_rad1,'rad2':saved_rad2}

@jit
def reaction_step(c,h,n,v,a,dt,D_c,D_h_mul,rho,beta_mul,gamma,alpha_h_mul,alpha_n,D_v,D_a,delta_c,delta_h,mu,q,lam,K,Km,P_cw,P_hw,P_cg,P_hg,v0,u0):
    for i in range(c.size):
        #print(f'------------{i}------------')
        Qn=np.array([c[i],h[i],n[i],v[i],a[i]])
        if np.linalg.norm(Qn)==0:
            continue

        # Stage 1: TR
        Rn=np.array(reaction(Qn[0],Qn[1],Qn[2],Qn[3],Qn[4],D_c,D_h_mul,rho,beta_mul,gamma,alpha_h_mul,alpha_n,D_v,D_a,delta_c,delta_h,mu,q,lam,K,Km,P_cw,P_hw,P_cg,P_hg,v0,u0))
        M=Qn+dt/4 * Rn
        Qs=Qn.copy()

        for _ in range(1000):
            Rs=np.array(reaction(Qs[0],Qs[1],Qs[2],Qs[3],Qs[4],D_c,D_h_mul,rho,beta_mul,gamma,alpha_h_mul,alpha_n,D_v,D_a,delta_c,delta_h,mu,q,lam,K,Km,P_cw,P_hw,P_cg,P_hg,v0,u0))
            Js=numerical_jacobian_jit(Qs,D_c,D_h_mul,rho,beta_mul,gamma,alpha_h_mul,alpha_n,D_v,D_a,delta_c,delta_h,mu,q,lam,K,Km,P_cw,P_hw,P_cg,P_hg,v0,u0)
            dQ=np.linalg.solve(np.eye(5)-dt/4 * Js, -(Qs-dt/4 * Rs-M))
            Qs+=dQ
            if np.linalg.norm(dQ)<1e-4:
                break
        
        #Stage 2: BDF2
        M=(4*Qs-Qn)/3
        Qnp1=Qs.copy()

        for _ in range(20):
            Rnp=np.array(reaction(Qnp1[0],Qnp1[1],Qnp1[2],Qnp1[3],Qnp1[4],D_c,D_h_mul,rho,beta_mul,gamma,alpha_h_mul,alpha_n,D_v,D_a,delta_c,delta_h,mu,q,lam,K,Km,P_cw,P_hw,P_cg,P_hg,v0,u0))
            Jnp=numerical_jacobian_jit(Qnp1,D_c,D_h_mul,rho,beta_mul,gamma,alpha_h_mul,alpha_n,D_v,D_a,delta_c,delta_h,mu,q,lam,K,Km,P_cw,P_hw,P_cg,P_hg,v0,u0)
            dQ=np.linalg.solve(np.eye(5)-dt/3 * Jnp, -(Qnp1-dt/3 * Rnp - M))
            Qnp1+=dQ
            if np.linalg.norm(dQ)<1e-4:
                break
        
        c[i],h[i],n[i],v[i],a[i]=Qnp1
    return c,h,n,v,a