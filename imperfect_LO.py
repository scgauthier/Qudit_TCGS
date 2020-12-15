import time
import os.path
import numpy as np
from itertools import product
from cmath import exp
from math import pi
from utils import getbasisR,get_GHZ_adj,get_lin_adj,assign_nodes

#********VARIABLES********************************************************#
#dim: dimension of state_shape
#num_nodes: number of nodes in graph state
#graph_type: options are 'GHZ', 'line'. Specifies type of graph state by its adjacency matrix.
#operation: options are 'raise' or 'lower'. Defines whether controlled two qudit gate applies X or X^(d-1)
#indA1 (indB1) [indA2] {indB2}: row (col) [row] {col} associated with current
#    coefficient from 2 state density matrix
#target_node: qudit on which gate should act
#coef_mat: stores all the information about the current density matrix
#set: options--sA or sB--describes whether gate acts on subset of nodes A or B
#**************************************************************************#

#**************************************************************************#
#defines a noisy controlled two qudit operation, which can either be a raising
#operation or a lowering operation
#**************************************************************************#
def perfect_CG(dim,num_nodes,graph_type,operation,indA1,indB1,indA2,indB2, target_node,coef_mat,set):

    numA,numB=assign_nodes(num_nodes,graph_type)

    if graph_type=='GHZ':
        adj_mat=get_GHZ_adj(num_nodes)
    elif graph_type=='line':
        adj_mat=get_lin_adj(num_nodes)
    else:
        print('Supported graph types are GHZ and line')
        return
    Alabels=list(product(np.arange(0,dim),repeat=numA))
    Blabels=list(product(np.arange(0,dim),repeat=numB))

    #get labels of state 1 entry and state 2 entry
    lab1A=np.array(Alabels[indA1])
    lab2A=np.array(Alabels[indA2])
    lab1B=np.array(Blabels[indB1])
    lab2B=np.array(Blabels[indB2])

    if set=='sA': #Node set on which operation acts
        outputs=[] #These will be packaged as [coef,outLab1B,outLab2A]--raise or [coef,outLab1A,outLab2B]--lower
        cutAdjRow = adj_mat[target_node-1,numA:(numA+numB)]
        cutBasisVec=getbasisR(numA+numB)[target_node-1][0:numA]

        if operation=='raise': #For perfect cLower
            for n in range(1,dim):
                outLab1B = (lab1B + n*(dim-1)*cutAdjRow) % dim
                for m in range(1,dim):
                    outLab2A = (lab2A + m*cutBasisVec) % dim
                    coef=(1/dim)*exp((2*pi*complex(0,1)/dim)*((n*(dim-1)*lab1A[target_node-1] + (dim-m*n)) % dim))
                    outputs.append([coef,outLab1B,outLab2A])
                    print('package: ', [coef,outLab1B,outLab2A])

        elif operation=='lower': #For perfect cLower
            for n in range(1,dim):
                outLab2B = (lab2B + n*cutAdjRow) % dim
                for m in range(1,dim):
                    outLab1A = (lab1A + m*cutBasisVec) % dim
                    coef=(1/dim)*exp((2*pi*complex(0,1)/dim)*((n*lab2A[target_node-1] + (dim-m*n)) % dim))
                    outputs.append([coef,outLab1A,outLab2B])
                    print('package: ', [coef, outLab1A,outLab2B])


    elif set=='sB': #Node set on which operation acts
        outputs=[] #These will be packaged as [coef,outLab1B,outLab2A]--raise or [coef, outLab1B,outLab2A]--lower
        cutAdjRow = adj_mat[target_node-1,0:numA]
        cutBasisVec=getbasisR(numA+numB)[target_node-1][numA:(numA+numB)]

        if operation=='raise': #For perfect cRaise:
            for n in range(1,dim):
                outLab1A = (lab1A + n*(dim-1)*cutAdjRow) % dim
                for m in range(1,dim):
                    outLab2B = (lab2B + m*cutBasisVec) % dim
                    coef=(1/dim)*exp((2*pi*complex(0,1)/dim)*((n*(dim-1)*lab1B[target_node-1-numA] + (dim-m*n)) % dim))
                    outputs.append([coef,outLab1A,outLab2B])
                    print('package: ', [coef,outLab1A,outLab2B])

        elif operation=='lower': #For perfect cLower
            for n in range(1,dim):
                outLab2A = (lab2A + n*cutAdjRow) % dim
                for m in range(1,dim):
                    outLab1B=(lab1B + m*cutBasisVec) % dim
                    coef=(1/dim)*exp((2*pi*complex(0,1)/dim)*((n*(dim-1)*lab2B[target_node-1-numA] + (dim-m*n)) % dim ))
                    outputs.append([coef,outLab1B,outLab2A])
                    print('package: ', [coef,outLab1B,outLab2A])
    else:
        print('Supported sets are sA or sB')
        return




#**************************************************************************#
cm=np.zeros((8,4))
cm[0,0]=1
perfect_CG(2,5,'line','lower',0,1,6,1,4,cm,'sB')

# print(get_lin_adj(5))
