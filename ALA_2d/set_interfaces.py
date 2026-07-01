import numpy as np

def set_interfaces(Slice,P):
    n1,n2=Slice.shape
    white_mask=(Slice==0)
    grey_mask=(Slice==1)
    Dweight=np.zeros_like(Slice,dtype=np.float64)
    Dweight[grey_mask]=1
    Dweight[white_mask]=P['R_wg']

    flat=Dweight.ravel(order='F')
    ind_nz=np.flatnonzero(flat>0)
    nz_set=set(ind_nz)

    #Boundary nodes
    boundary=[]
    for I in ind_nz:
        neighbours=[I-1,I+1,I-n1,I+n1]
        for nb in neighbours:
            if nb<0 or nb>=n1*n2 or (nb not in nz_set): #just the (nb not in nz_set)?
                boundary.append(I)
                break
    boundary=np.array(boundary,dtype=np.uint32)

    #Downwards
    Origin1=[]
    Terminus1=[]
    Dweight1=[]
    for I in ind_nz:
        if ((I+1)%n1)!=0:
            J=I+1
            if J<n1*n2 and J in nz_set:
                Origin1.append(I)
                Terminus1.append(J)
                Dweight1.append(0.5*(flat[I]+flat[J]))
    
    Origin1=np.array(Origin1)
    Terminus1=np.array(Terminus1)
    Dweight1=np.array(Dweight1)

    #Across
    Origin2=[]
    Terminus2=[]
    Dweight2=[]
    for I in ind_nz:
        if ((I+n1)%n2)!=0:
            J=I+n1
            if J<n1*n2 and J in nz_set:
                Origin2.append(I)
                Terminus2.append(J)
                Dweight2.append(0.5*(flat[I]+flat[J]))
    
    Origin2=np.array(Origin2)
    Terminus2=np.array(Terminus2)
    Dweight2=np.array(Dweight2)

    return (Dweight1,Dweight2,Origin1,Origin2,Terminus1,Terminus2,ind_nz,boundary)