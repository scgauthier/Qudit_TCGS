import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from operator import add

#num_nodes: number of nodes in graph
#dim: dimension of qudit
#graph_type: GHZ, line
#input_type: oneParam,

start_time=time.time()

def assign_nodes(num_nodes,graph_type):
    if graph_type == 'GHZ':
        numA=1
        numB=num_nodes-1
    elif graph_type =='line':
        if (num_nodes % 2)==0:
            numA=num_nodes/2
            numB=num_nodes/2
        else:
            numA=(num_nodes+1)/2
            numB=(num_nodes-1)/2
    elif graph_type =='cluster':
        numA=num_nodes/2
        numB=num_nodes/2
    return numA,numB

def get_input_coefficients(num_nodes,dim,graph_type,input_type,param):
    numA,numB=assign_nodes(num_nodes,graph_type)

    #rows associated with A labels
    #columns with B labels
    coef_mat=np.zeros((dim**numA,dim**numB))

    #set coefficients
    if input_type=='oneParam':
        coef_mat[0,0]= param+(1-param)/(dim**num_nodes)

        for x in range(0,dim**numA):
            for y in range(0,dim**numB):
                if x==0 and y==0:
                    continue
                else:
                    coef_mat[x,y]=(1-param)/(dim**num_nodes)

    return coef_mat

def P1_update_coefficients(num_nodes,dim,graph_type,cmat_in):
    dim_list=list(range(0,dim))

    numA,numB=assign_nodes(num_nodes,graph_type)

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


    return coef_mat


coef_mat=get_input_coefficients(3,3,'GHZ','oneParam',0.6)
#print(coef_mat)
print(P1_update_coefficients(3,3,'GHZ',coef_mat))


print("--- %s seconds ---" % (time.time()-start_time))
