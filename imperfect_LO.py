import time
import os.path
import numpy as np
from itertools import product
from cmath import exp
from math import pi
from utils import getbasisR,get_GHZ_adj,get_lin_adj,assign_nodes,index_convert
from utils import match_labA_to_indA,match_labB_to_indB,get_two_state_indices
from weyl_covariant_channel import qudit_through_channel

#********VARIABLES********************************************************#
#dim: dimension of state_shape
#num_nodes: number of nodes in graph state
#graph_type: options are 'GHZ', 'line'. Specifies type of graph state by its adjacency matrix.
#operation: options are 'raise' or 'lower'. Defines whether controlled two qudit gate applies X or X^(d-1)
#row (col): row (column) associated with current entry of the two state coefficient matrix
#    coefficient from 2 state density matrix
#target_node: qudit on which gate should act
#coef_mat: stores all the information about the current density matrix
#set: options--sA or sB--describes whether gate acts on subset of nodes A or B
#**************************************************************************#

#**************************************************************************#
#Helper function to do the Craise operation on set A qudit (direction 21)
#**************************************************************************#
def raiseA(dim,numA,numB,row,col,target_node,adj_mat,new_coef_mat,current_coef):

    cutAdjRow = adj_mat[target_node-1,numA:(numA+numB)]
    cutBasisVec=getbasisR(numA+numB)[target_node-1][0:numA]

    indA1,indA2,indB1,indB2=get_two_state_indices(dim,numA,numB,row,col)
    labB1=list(product(np.arange(dim),repeat=numB))[indB1]
    labA2=list(product(np.arange(dim),repeat=numA))[indA2]
    labA1=list(product(np.arange(dim),repeat=numA))[indA1]

    for n in range(dim):
         outLabB1 = (labB1 + n*(dim-1)*cutAdjRow) % dim
         indB1=match_labB_to_indB(dim,numB,outLabB1)
         for m in range(0,dim):
             outLabA2 = (labA2 + m*cutBasisVec) % dim
             indA2=match_labA_to_indA(dim,numA,outLabA2)
             cf=(1/dim)*exp((2*pi*complex(0,1)/dim)*((n*(dim-1)*labA1[target_node-1] + (dim-m*n)) % dim))

             new_row,new_col=index_convert(dim,numA,numB,indA1,indB1,indA2,indB2)
             new_coef_mat[new_row,new_col]+=cf*current_coef

    return new_coef_mat

#**************************************************************************#
#Helper function to do the Clower operation on set A qudit (direction 12)
#**************************************************************************#
def lowerA(dim,numA,numB,labA1,labA2,labB2,indB1,indA2,target_node,adj_mat,new_coef_mat,current_coef):

    cutAdjRow = adj_mat[target_node-1,numA:(numA+numB)]
    cutBasisVec=getbasisR(numA+numB)[target_node-1][0:numA]

    indA1,indA2,indB1,indB2=get_two_state_indices(dim,numA,numB,row,col)
    labB2=list(product(np.arange(dim),repeat=numB))[indB2]
    labA2=list(product(np.arange(dim),repeat=numA))[indA2]
    labA1=list(product(np.arange(dim),repeat=numA))[indA1]

    for n in range(dim):
        outLabB2 = (labB2 + n*cutAdjRow) % dim
        indB2=match_labB_to_indB(dim,numB,outLabB2)
        for m in range(0,dim):
            outLabA1 = (labA1 + m*cutBasisVec) % dim
            indA1=match_labA_to_indA(dim,numA,outLabA1)
            cf=(1/dim)*exp((2*pi*complex(0,1)/dim)*((n*labA2[target_node-1] + (dim-m*n)) % dim))

            new_row,new_col=index_convert(dim,numA,numB,indA1,indB1,indA2,indB2)
            new_coef_mat[new_row,new_col]+=cf*current_coef

    return new_coef_mat

#**************************************************************************#
#Helper function to do the Clower operation on set B qudit (direction 12)
#**************************************************************************#
def lowerB(dim,numA,numB,row,col,target_node,adj_mat,new_coef_mat,current_coef):

    cutAdjRow = adj_mat[target_node-1,0:numA]
    cutBasisVec=getbasisR(numA+numB)[target_node-1][numA:(numA+numB)]

    indA1,indA2,indB1,indB2=get_two_state_indices(dim,numA,numB,row,col)
    labB2=list(product(np.arange(dim),repeat=numB))[indB2]
    labB1=list(product(np.arange(dim),repeat=numB))[indB1]
    labA2=list(product(np.arange(dim),repeat=numA))[indA2]

    for n in range(0,dim):
        outLabA2 = (labA2 + n*cutAdjRow) % dim
        indA2=match_labA_to_indA(dim,numA,outLabA2)
        for m in range(0,dim):
            outLabB1=(labB1 + m*cutBasisVec) % dim
            indB1=match_labB_to_indB(dim,numB,outLabB1)
            cf=(1/dim)*exp((2*pi*complex(0,1)/dim)*((n*labB2[target_node-1-numA] + (dim-m*n)) % dim ))

            new_row,new_col=index_convert(dim,numA,numB,indA1,indB1,indA2,indB2)
            new_coef_mat[new_row,new_col]+=cf*current_coef

    return new_coef_mat

#**************************************************************************#
#Helper function to do the Craise operation on set B qudit (direction 21)
#**************************************************************************#
def raiseB(dim,numA,numB,labA1,labB1,labB2,indB1,indA2,target_node,adj_mat,new_coef_mat,current_coef):

    cutAdjRow = adj_mat[target_node-1,0:numA]
    cutBasisVec=getbasisR(numA+numB)[target_node-1][numA:(numA+numB)]

    indA1,indA2,indB1,indB2=get_two_state_indices(dim,numA,numB,row,col)
    labB2=list(product(np.arange(dim),repeat=numB))[indB2]
    labB1=list(product(np.arange(dim),repeat=numB))[indB1]
    labA1=list(product(np.arange(dim),repeat=numA))[indA1]

    for n in range(0,dim):
        outLabA1 = (labA1 + n*(dim-1)*cutAdjRow) % dim
        indA1=match_labA_to_indA(dim,numA,outLabA1)
        for m in range(0,dim):
            outLabB2 = (labB2 + m*cutBasisVec) % dim
            indB2=match_labB_to_indB(dim,numB,outLabB2)
            cf=(1/dim)*exp((2*pi*complex(0,1)/dim)*((n*(dim-1)*labB1[target_node-1-numA] + (dim-m*n)) % dim))

            new_row,new_col=index_convert(dim,numA,numB,indA1,indB1,indA2,indB2)
            new_coef_mat[new_row,new_col]+=cf*current_coef

    return new_coef_mat

#**************************************************************************#
#defines a noisy controlled two qudit operation, which can either be a raising
#operation or a lowering operation
#**************************************************************************#
def perfect_CG(dim,num_nodes,graph_type,operation,target_node,coef_mat,set):

    numA,numB=assign_nodes(num_nodes,graph_type) #partition nodes
    #get adj_mat
    if graph_type=='GHZ':
        adj_mat=get_GHZ_adj(num_nodes)
    elif graph_type=='line':
        adj_mat=get_lin_adj(num_nodes)
    else:
        print('Supported graph types are GHZ and line')
        return
    #get a clean slate matrix for gate outcomes
    cm=complex(1,0)*np.zeros((dim**(2*numA),dim**(2*numB)))

    #decide which operation to do
    if set=='sA':
        if operation=='raise':
            for i in range(dim**(2*numA)):
                for j in range(dim**(2*numB)):
                    if abs(coef_mat[i,j])>1e-14:
                        cm=raiseA(dim,numA,numB,i,j,target_node,adj_mat,cm,coef_mat[i,j])
        elif operation=='lower':
            for i in range(dim**(2*numA)):
                for j in range(dim**(2*numB)):
                    if abs(coef_mat[i,j])>1e-14:
                        cm=lowerA(dim,numA,numB,i,j,target_node,adj_mat,cm,coef_mat[i,j])
        else:
            print('Supported operations are raise and lower')
            return

    elif set=='sB':
        if operation=='raise':
            for i in range(dim**(2*numA)):
                for j in range(dim**(2*numB)):
                    if abs(coef_mat[i,j])>1e-14:
                        cm=raiseB(dim,numA,numB,i,j,target_node,adj_mat,cm,coef_mat[i,j])
        elif operation=='lower':
            for i in range(dim**(2*numA)):
                for j in range(dim**(2*numB)):
                    if abs(coef_mat[i,j])>1e-14:
                        cm=lowerB(dim,numA,numB,i,j,target_node,adj_mat,cm,coef_mat[i,j])
        else:
            print('Supported operations are raise and lower')
            return

    else:
        print('Supported sets are sA and sB')
        return

    return cm

#**************************************************************************#
#Defines noisy two qudit operation. First each qudit that the gate will act
#on is sent through a depolarizing channel, then the perfect gate is performed
#**************************************************************************#
def noisy_TQG():
    #first: run both qudits involved in gate through depolarizing Channel
    #second: do two perfect two qudit gate
    return pi


#**************************************************************************#

zm=complex(1,0)*np.zeros((9,81))
zm[0,0]=1
coef_mat=perfect_CG(3,3,'GHZ','raise',1,zm,'sA')
coef_mat=perfect_CG(3,3,'GHZ','lower',2,coef_mat,'sB')
coef_mat=perfect_CG(3,3,'GHZ','lower',3,coef_mat,'sB')
print(coef_mat)
# print(get_lin_adj(5))
