import time
import os.path
import numpy as np
from itertools import product


#**************************************************************************#
#Runs sub-protocol P1 on a given input state, specified by cmat_in and the
#type params dim, num_nodes, graph_type
#**************************************************************************#
def P1_update_coefficients(num_nodes,dim,graph_type,cmat_in):
    dim_list=list(range(0,dim))

    numA,numB=assign_nodes(num_nodes,graph_type) #determines node bi-partition

    #Calculate normalization coefficient
    normK=0
    for x in range(0,dim**numA):
        for y in range(0,dim**numB):
            for z in range(0,y):
                normK+=cmat_in[x,y]*cmat_in[x,z]+cmat_in[x,z]*cmat_in[x,y]
            normK+=(cmat_in[x,y])**2

    indices=list(product(dim_list,repeat=numB))
    coef_mat=np.zeros((dim**numA,dim**numB))
    for control in range(0,dim**numB):
        indexControl=np.array(indices[control])
        # good_entries=[]
        for y in range(0,dim**numB):
            indexB1=np.array(indices[y])

            for z in range(0,y+1):
                indexB2=np.array(indices[z])

                for entry in range(0,numB):
                    if ((indexB1+indexB2+(dim-1)*indexControl) % dim)[numB-1-entry]!=0:
                        break
                    elif entry==numB-1 and y!=z:
                        for x in range(0,dim**numA):
                            coef_mat[x,control]+=(cmat_in[x,y]*cmat_in[x,z]+cmat_in[x,z]*cmat_in[x,y])/normK
                    elif entry==numB-1 and y==z:
                        for x in range(0,dim**numA):
                            coef_mat[x,control]+=((cmat_in[x,y])**2)/normK


    return normK,coef_mat
