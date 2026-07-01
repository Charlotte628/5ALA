import numpy as np
from numba import jit

@jit
def reaction(c,h,ao,ah,ac,p,po,hi,D_ao,k_h,k_c,k_mh,k_mc,r_h,r_c,r_2,k_m2,k_2,D_p,r_3):
    dao=-k_h*h*ao+k_mh*ah-k_c*c*ao+k_mc*ac
    dah=k_h*h*ao-r_h*ah-k_mh*ah
    dac=k_c*c*ao-(r_c+k_mc)*ac
    dp=r_h*ah+r_c*ac-r_2*p-k_m2*p+k_2*(c+h)*po
    dpo=k_m2*p-k_2*(c+h)*po
    dhi=r_2*p-r_3*hi

    return dao,dah,dac,dp,dpo,dhi