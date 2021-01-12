import time
import os.path
import numpy as np
from itertools import product
from random import randint
from cmath import exp
from math import pi
from utils import getbasisR,get_GHZ_adj,get_lin_adj,assign_nodes,index_convert
from utils import match_labA_to_indA,match_labB_to_indB,get_two_state_indices
from utils import compare_labels,label_update,match_label_to_index, calc_reduced_state
# from weyl_covariant_channel import qudit_through_channel

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
#state_copy: options--first or second--describes whether depolarizing channel acts
#           on qudit from first or second state copy
#target_state: options--first or second--do partial trace to get reduced first or second state
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
         indB1=match_labB_to_indB(dim,outLabB1)
         for m in range(0,dim):
             outLabA2 = (labA2 + m*cutBasisVec) % dim
             indA2=match_labA_to_indA(dim,outLabA2)
             cf=(1/dim)*exp((2*pi*complex(0,1)/dim)*((n*(dim-1)*labA1[target_node-1] + (dim-m*n)) % dim))

             new_row,new_col=index_convert(dim,numA,numB,indA1,indB1,indA2,indB2)
             new_coef_mat[new_row,new_col]+=cf*current_coef

    return new_coef_mat

#**************************************************************************#
#Helper function to do the Clower operation on set A qudit (direction 12)
#**************************************************************************#
def lowerA(dim,numA,numB,row,col,target_node,adj_mat,new_coef_mat,current_coef):

    cutAdjRow = adj_mat[target_node-1,numA:(numA+numB)]
    cutBasisVec=getbasisR(numA+numB)[target_node-1][0:numA]

    indA1,indA2,indB1,indB2=get_two_state_indices(dim,numA,numB,row,col)
    labB2=list(product(np.arange(dim),repeat=numB))[indB2]
    labA2=list(product(np.arange(dim),repeat=numA))[indA2]
    labA1=list(product(np.arange(dim),repeat=numA))[indA1]

    for n in range(dim):
        outLabB2 = (labB2 + n*cutAdjRow) % dim
        indB2=match_labB_to_indB(dim,outLabB2)
        for m in range(0,dim):
            outLabA1 = (labA1 + m*cutBasisVec) % dim
            indA1=match_labA_to_indA(dim,outLabA1)
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
        indA2=match_labA_to_indA(dim,outLabA2)
        for m in range(0,dim):
            outLabB1=(labB1 + m*cutBasisVec) % dim
            indB1=match_labB_to_indB(dim,outLabB1)
            cf=(1/dim)*exp((2*pi*complex(0,1)/dim)*((n*labB2[target_node-1-numA] + (dim-m*n)) % dim ))

            new_row,new_col=index_convert(dim,numA,numB,indA1,indB1,indA2,indB2)
            new_coef_mat[new_row,new_col]+=cf*current_coef

    return new_coef_mat

#**************************************************************************#
#Helper function to do the Craise operation on set B qudit (direction 21)
#**************************************************************************#
def raiseB(dim,numA,numB,row,col,target_node,adj_mat,new_coef_mat,current_coef):

    cutAdjRow = adj_mat[target_node-1,0:numA]
    cutBasisVec=getbasisR(numA+numB)[target_node-1][numA:(numA+numB)]

    indA1,indA2,indB1,indB2=get_two_state_indices(dim,numA,numB,row,col)
    labB2=list(product(np.arange(dim),repeat=numB))[indB2]
    labB1=list(product(np.arange(dim),repeat=numB))[indB1]
    labA1=list(product(np.arange(dim),repeat=numA))[indA1]

    for n in range(0,dim):
        outLabA1 = (labA1 + n*(dim-1)*cutAdjRow) % dim
        indA1=match_labA_to_indA(dim,outLabA1)
        for m in range(0,dim):
            outLabB2 = (labB2 + m*cutBasisVec) % dim
            indB2=match_labB_to_indB(dim,outLabB2)
            cf=(1/dim)*exp((2*pi*complex(0,1)/dim)*((n*(dim-1)*labB1[target_node-1-numA] + (dim-m*n)) % dim))

            new_row,new_col=index_convert(dim,numA,numB,indA1,indB1,indA2,indB2)
            new_coef_mat[new_row,new_col]+=cf*current_coef

    return new_coef_mat

#**************************************************************************#
#defines a controlled two qudit operation, which can either be a raising
#operation or a lowering operation
#**************************************************************************#
def perfect_CG(dim,numA,numB,adj_mat,operation,target_node,coef_mat,set):

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


#***************************************************************************#
def qudit_through_channel(dim,numA,numB,adj_mat,target_node,state_copy,coef_mat,param):

    input_coef_mat=np.copy(coef_mat)
    new_coef_mat=(1-param)*np.copy(coef_mat)

    Alabels=list(product(np.arange(0,dim),repeat=numA))
    Blabels=list(product(np.arange(0,dim),repeat=numB))

    for x in range(dim**2):
        error_label=list(product(np.arange(0,dim),repeat=2))[x]

        for row in range(dim**(2*numA)):
            for col in range(dim**(2*numB)):

                indA1,indA2,indB1,indB2=get_two_state_indices(dim,numA,numB,row,col)
                current_coef=input_coef_mat[row,col]

                if state_copy=='first':
                    if current_coef>0:
                        labelIn=np.array(Alabels[indA1]+Blabels[indB1])

                        #create list of effected labels
                        altered=[]

                        labelOut=label_update(dim,adj_mat,target_node,labelIn,error_label)

                        if compare_labels(labelOut,labelIn):
                            altered.append(labelOut)
                            #print('altered',altered)

                    #determine which coefficients need to change and change them
                    for entry in range(np.shape(altered)[0]):
                        #graphs state basis label of coefficient to be updated
                        focus=altered[entry]
                        #get matrix indices of label
                        id_indA1,id_indB1=match_label_to_index(dim,numA,numB,focus)
                        #convert back to row and col of new_coef_mat
                        id_row,id_col=index_convert(dim,numA,numB,id_indA1,id_indB1,indA2,indB2)
                        #update coeficient matrix
                        new_coef_mat[id_row,id_col]+=current_coef*(param/(dim**2 -1))

                elif state_copy=='second':
                    if current_coef>0:
                        labelIn=np.array(Alabels[indA2]+Blabels[indB2])

                        #create list of effected labels
                        altered=[]

                        labelOut=label_update(dim,adj_mat,target_node,labelIn,error_label)

                        if compare_labels(labelOut,labelIn):
                            altered.append(labelOut)
                            #print('altered',altered)

                    #determine which coefficients need to change and change them
                    for entry in range(np.shape(altered)[0]):
                        #graphs state basis label of coefficient to be updated
                        focus=altered[entry]
                        #get matrix indices of label
                        id_indA2,id_indB2=match_label_to_index(dim,numA,numB,focus)
                        #convert back to row and col of new_coef_mat
                        id_row,id_col=index_convert(dim,numA,numB,indA1,indB1,id_indA2,id_indB2)
                        #update coeficient matrix
                        new_coef_mat[id_row,id_col]+=current_coef*(param/(dim**2 -1))

    return new_coef_mat

#**************************************************************************#
#Obtain random measurement results and make sure they satisfy the post-selection
#condition, if they do not then obtain new measurement results and keep trying
#**************************************************************************#
def measure_and_post_select(dim,numA,numB,adj_mat):
    dim_list=np.arange(dim)
    #All possible measurement results
    Alabels=list(product(dim_list,repeat=numA))
    Blabels=list(product(dim_list,repeat=numB))

    unmeasured=True
    while unmeasured==True:
        #choose random measurement result
        jB=np.array(Blabels[randint(0,dim**numB - 1)])
        jA=np.array(Alabels[randint(0,dim**numA - 1)])

        #check if measurement condition is satisfied
        for x in range(numA):
            checkSum=0
            #the B part of the sum
            for y in range(numB):
                checkSum+=adj_mat[x,numA+y]*jB[y]
            #the A part of the sum
            checkSum+=jA[x]
            if (checkSum % dim)!=0:
                unmeasured=True
                break
            unmeasured=False
    return jA,jB

#**************************************************************************#
#Following obtaining a successful measurement label, update the coefficient
#matrix appropriately
#**************************************************************************#
def post_measurement_state_update(dim,numA,numB,cmat_in,csubP):

    dim_list=np.arange(dim)
    new_coef_mat=complex(1,0)*np.zeros((dim**numA,dim**numB))

    normK=0

    if csubP=='P1':
        indA2=0
        #Application of measurement condition
        for indA1 in range(dim**numA): #fix row of updated state
            for indB1 in range(dim**numB): #fix col of updated state
                for indB2 in range(dim**numB): #loop through other coefficients
                    row,col=index_convert(dim,numA,numB,indA1,indB1,indA2,indB2)
                    new_coef_mat[indA1,indB1]+=cmat_in[row,col]
                    normK+=cmat_in[row,col] #calculate normK while updating
        new_coef_mat=new_coef_mat/normK

    elif csubP=='P2':
        indB2=0
        #Application of measurement condition
        for indA1 in range(dim**numA): #fix row of updated state
            for indB1 in range(dim**numB): #fix col of updated state
                for indA2 in range(dim**numA): #loop through other coefficients
                    row,col=index_convert(dim,numA,numB,indA1,indB1,indA2,indB2)
                    new_coef_mat[indA1,indB1]+=cmat_in[row,col]
                    normK+=cmat_in[row,col] #calculate normK while updating
        new_coef_mat=new_coef_mat/normK

    return new_coef_mat

#**************************************************************************#
#Defines noisy two qudit operation. First each qudit that the gate will act
#on is sent through a depolarizing channel, then the perfect gate is performed
#**************************************************************************#
def noisy_TQG(dim,numA,numB,adj_mat,operation,target_node,coef_mat,set,param):

    #apply noise to qudit from first copy of state
    coef_mat=qudit_through_channel(dim,numA,numB,adj_mat,target_node,'first',coef_mat,param)
    #apply noise to qudit from second copy of state
    coef_mat=qudit_through_channel(dim,numA,numB,adj_mat,target_node,'second',coef_mat,param)
    #do perfect gate between the two qudits
    coef_mat=perfect_CG(dim,numA,numB,adj_mat,operation,target_node,coef_mat,set)

    return coef_mat

#**************************************************************************#
#Runs version of sub-protocol P1 with noisy two qudit operations on a given
#input state, specified by cmat_in and the type params dim, num_nodes, graph_type
#**************************************************************************#
def noisy_protocol_P1(num_nodes,dim,graph_type,cmat,param):

    #turn one state input into two state input
    cmat=np.kron(cmat,cmat)
    numA,numB=assign_nodes(num_nodes,graph_type) #determines node bi-partition
    #get adj_mat
    if graph_type=='GHZ':
        adj_mat=get_GHZ_adj(num_nodes)
    elif graph_type=='line':
        adj_mat=get_lin_adj(num_nodes)
    else:
        print('Supported graph types are GHZ and line')
        return

    #do all noisy gates of P1
    for x in range(1,num_nodes+1):
        if x<=numA:
            set='sA'
            operation='raise'
        else:
            set='sB'
            operation='lower'
        cmat=noisy_TQG(dim,numA,numB,adj_mat,operation,x,cmat,set,param)

    #'measure' & 'post-select' --> Use the coefficient map for general states
    cmat=post_measurement_state_update(dim,numA,numB,cmat,'P1') #input 2 states, output 1 state

    return cmat
#**************************************************************************#
#Runs version of subprotocol P2 with noisy two qudit operations on a given input state
#**************************************************************************#
def noisy_protocol_P2(num_nodes,dim,graph_type,cmat,param):

    #turn one state input into two state input
    cmat=np.kron(cmat,cmat)
    numA,numB=assign_nodes(num_nodes,graph_type) #determines node bi-partition
    #get adj_mat
    if graph_type=='GHZ':
        adj_mat=get_GHZ_adj(num_nodes)
    elif graph_type=='line':
        adj_mat=get_lin_adj(num_nodes)
    else:
        print('Supported graph types are GHZ and line')
        return

    #do all noisy gates of P1
    for x in range(1,num_nodes+1):
        if x<=numA:
            set='sA'
            operation='lower'
        else:
            set='sB'
            operation='raise'
        cmat=noisy_TQG(dim,numA,numB,adj_mat,operation,x,cmat,set,param)

    #'measure' & 'post-select' --> Use the coefficient map for general states
    cmat=post_measurement_state_update(dim,numA,numB,cmat,'P2') #input 2 states, output 1 state

    return cmat


#**************************************************************************#
# numA=1
# numB=2
# dim=3
# graph_type='GHZ'
# zm=complex(1,0)*np.zeros((3,9))
# zm[0,0]=1
# print(zm, '\n')
# cmat=noisy_protocol_P2(numA+numB,dim,graph_type,zm,0.1)
# print(cmat, '\n')
