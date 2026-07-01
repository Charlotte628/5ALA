import numpy as np
from scipy.sparse.linalg import cg,LinearOperator
from typing import Dict,Any

from ALA_2d.diffusion_op import diffusion_op
from ALA_2d.reaction import reaction

from numba import jit

@jit
def numerical_jacobian_jit(Q,h,c,D_ao,k_h,k_c,k_mh,k_mc,r_h,r_c,r_2,k_m2,k_2,D_p,r_3):
    J=np.zeros((6,6))
    ao=Q[0]
    ah=Q[1]
    ac=Q[2]
    p=Q[3]
    po=Q[4]
    hi=Q[5]

    J[0,0]=-k_h*h-k_c*c #ao,ao
    J[0,1]=k_mh #ao,ah
    J[0,2]=k_mc #ao,ac
    J[0,3]=0 #ao,p
    J[0,4]=0 #ao,po
    J[0,5]=0 #ao,hi

    J[1,0]=k_h*h #ah,ao
    J[1,1]=-r_h-k_mh #ah,ah
    J[1,2]=0 #ah,ac
    J[1,3]=0 #ah,p
    J[1,4]=0 #ah,po
    J[1,5]=0 #ah,hi

    J[2,0]=k_c*c #ac,ao
    J[2,1]=0 #ac,ah
    J[2,2]=-r_c-k_mc #ac,ac
    J[2,3]=0 #ac,p
    J[2,4]=0 #ac,po
    J[2,5]=0 #ac,hi

    J[3,0]=0 #p,ao
    J[3,1]=r_h #p,ah
    J[3,2]=r_c #p,ac
    J[3,3]=-r_2-k_m2 #p,p
    J[3,4]=-k_2*(c+h) #p,po
    J[3,5]=0 #p,hi

    J[4,0]=0 #po,ao
    J[4,1]=0 #po,ah
    J[4,2]=0 #po,ac
    J[4,3]=k_m2 #po,p
    J[4,4]=-k_2*(c+h) #po,po
    J[4,5]=0 #po,hi

    J[5,0]=0 #hi,ao
    J[5,1]=0 #hi,ah
    J[5,2]=0 #hi,ac
    J[5,3]=r_2 #hi,p
    J[5,4]=0 #hi,po
    J[5,5]=-r_3 #hi,hi

    return J 

def numerics_2d(
        Dweight1,Dweight2,
        Origin1,Origin2,
        Terminus1,Terminus2,
        n1,n2,ind_nz,boundary,
        IC,dayspersave, t_finish,
        radiussave, radius_final,
        P,c,h):
    dt=float(P['dt'])
    dx=float(P['dx'])

    D_ao=float(P['D_ao'])
    D_p=float(P['D_p'])
    k_h=float(P['k_h'])
    k_c=float(P['k_c'])
    k_mh=float(P['k_mh'])
    k_mc=float(P['k_mc'])
    r_h=float(P['r_h'])
    r_c=float(P['r_c'])
    r_2=float(P['r_2'])
    k_2=float(P['k_2'])
    k_m2=float(P['k_m2'])
    r_3=float(P['r_3'])

    ao=IC['ao0'].ravel(order='F').copy().astype(float)
    ah=IC['ah0'].ravel(order='F').copy().astype(float)
    ac=IC['ac0'].ravel(order='F').copy().astype(float)
    p=IC['p0'].ravel(order='F').copy().astype(float)
    po=IC['po0'].ravel(order='F').copy().astype(float)
    hi=IC['hi0'].ravel(order='F').copy().astype(float)

    N=n1*n2

    saved_times=[]
    saved_ao=[]
    saved_ah=[]
    saved_ac=[]
    saved_p=[]
    saved_po=[]
    saved_hi=[]

    def save_state(t):
        saved_times.append(t)
        saved_ao.append(ao.copy())
        saved_ah.append(ah.copy())
        saved_ac.append(ac.copy())
        saved_p.append(p.copy())
        saved_po.append(po.copy())
        saved_hi.append(hi.copy())

    t_dim=0
    t_step=0
    save_state(t_dim)

    steps_per_save=int(round(dayspersave)/dt)

    while True:
        print(f'--------{t_dim}--------')
        t_step+=1
        t_dim=t_step*dt

        N=ao.size
        A_ao=LinearOperator(shape=(N,N),matvec=lambda x:x-dt*diffusion_op(x,D_ao,dx,Dweight1,Dweight2,Origin1,Origin2,Terminus1,Terminus2))
        b_ao=dt*diffusion_op(ao,D_ao,dx,Dweight1,Dweight2,Origin1,Origin2,Terminus1,Terminus2)
        A_p=LinearOperator(shape=(N,N),matvec=lambda x:x-dt*diffusion_op(x,D_p,dx,Dweight1,Dweight2,Origin1,Origin2,Terminus1,Terminus2))
        b_p=dt*diffusion_op(ao,D_p,dx,Dweight1,Dweight2,Origin1,Origin2,Terminus1,Terminus2)

        dao,info=cg(A_ao,b_ao,rtol=1e-6,atol=0,maxiter=500)
        dp,info=cg(A_p,b_p,rtol=1e-6,atol=0,maxiter=500)

        ao=ao+dao
        p=p+dp

        print('Done Diffusion')

        ao=np.maximum(ao,0)
        p=np.maximum(p,0)
        ao,ah,ac,p,po,hi=reaction_step(dt,c,h,ao,ah,ac,p,po,hi,D_ao,k_h,k_c,k_mh,k_mc,r_h,r_c,r_2,k_m2,k_2,D_p,r_3)

        print('Done Reaction')

        if t_step%steps_per_save==0:
            save_state(t_dim)
        if t_dim>=t_finish-1e-12:
            break

    return {'time':saved_times,'ao':np.vstack(saved_ao),'ah':np.vstack(saved_ah),'ac':np.vstack(saved_ac),'p':np.vstack(saved_p),'po':np.vstack(saved_po),'hi':np.vstack(saved_hi)}

@jit
def reaction_step(dt,c,h,ao,ah,ac,p,po,hi,D_ao,k_h,k_c,k_mh,k_mc,r_h,r_c,r_2,k_m2,k_2,D_p,r_3):
    for i in range(ao.size):
        Qn=np.array([ao[i],ah[i],ac[i],p[i],po[i],hi[i]])
        if np.linalg.norm(Qn)==0:
            continue

        Rn=np.array(reaction(c[i],h[i],Qn[0],Qn[1],Qn[2],Qn[3],Qn[4],Qn[5],D_ao,k_h,k_c,k_mh,k_mc,r_h,r_c,r_2,k_m2,k_2,D_p,r_3))
        M=Qn+dt/4 * Rn
        Qs=Qn.copy()

        for _ in range(1000):
            Rs=np.array(reaction(c[i],h[i],Qs[0],Qs[1],Qs[2],Qs[3],Qs[4],Qs[5],D_ao,k_h,k_c,k_mh,k_mc,r_h,r_c,r_2,k_m2,k_2,D_p,r_3))
            Js=numerical_jacobian_jit(Qs,h[i],c[i],D_ao,k_h,k_c,k_mh,k_mc,r_h,r_c,r_2,k_m2,k_2,D_p,r_3)
            dQ=np.linalg.solve(np.eye(6)-dt/4 * Js, -(Qs-dt/4 * Rs-M))
            if np.linalg.norm(dQ)<1e-4:
                break

        M=(4*Qs-Qn)/3
        Qnp1=Qs.copy()

        for _ in range(20):
            Rnp=np.array(reaction(c[i],h[i],Qnp1[0],Qnp1[1],Qnp1[2],Qnp1[3],Qnp1[4],Qnp1[5],D_ao,k_h,k_c,k_mh,k_mc,r_h,r_c,r_2,k_m2,k_2,D_p,r_3))
            Jnp=numerical_jacobian_jit(Qnp1,h[i],c[i],D_ao,k_h,k_c,k_mh,k_mc,r_h,r_c,r_2,k_m2,k_2,D_p,r_3)
            dQ=np.linalg.solve(np.eye(6)-dt/3 * Jnp,-(Qnp1-dt/3 * Rnp - M))
            Qnp1+=dQ
            if np.linalg.norm(dQ)<1e-4:
                break
        
        ao[i],ah[i],ac[i],p[i],po[i],hi[i]=Qnp1
    return ao,ah,ac,p,po,hi