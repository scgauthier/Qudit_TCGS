import os.path
import numpy as np
from itertools import product
from quditgraphstates import get_GHZ_adj
from quditgraphstates import get_lin_adj

#********Weyl Channel Coefficients****************************************#
#we have made the parameter choice a_00=(1-q),
#a_uv=q/(dim^2 -1) forall u,v in (F_d)^2
#********VARIABLES********************************************************#

#dim: dimension of qudits
#graph_type: 'GHZ', 'line', 'cluster'
#adj_mat: adjacency matrix of graph state
#target_node: qudit being sent through channel
#label: a label of a graph basis state
#error label: describes a combination of X and Z errors to be applied
#coef_mat: fully characterizes the diagonal density matrix via a coefficient for each basis state
#param: transmission parameter of the channel
#kLab (unLab): an (un)known graph state basis labels

#**************************************************************************#
def assign_nodes(num_nodes,graph_type):
    if graph_type == 'GHZ':
        numA=1
        numB=num_nodes-1
    elif graph_type =='line':
        if (num_nodes % 2)==0:
            numA=int(num_nodes/2)
            numB=int(num_nodes/2)
        else:
            numA=int((num_nodes+1)/2)
            numB=int((num_nodes-1)/2)
    elif graph_type =='cluster':
        numA=int(num_nodes/2)
        numB=int(num_nodes/2)
    return numA,numB
#**************************************************************************#
def perfect_coef_mat(dim,numA,numB):
    coef_mat=np.zeros((dim**numA,dim**numB))
    coef_mat[0,0]=1
    return coef_mat
#**************************************************************************#
#Compares the labels of graph basis states to determine if they match
#returns True if they differ, False if the same
def compare_labels(unLab,kLab):
    for x in range(np.shape(kLab)[0]):
        if unLab[x]!=kLab[x]:
            return True
        elif x==(np.shape(kLab)[0]-1) and unLab[x]==kLab[x]:
            return False
#**************************************************************************#
#updates the graph state basis label associated after doing
#the specified combination of X and Z errors on the target_qudit
def label_update(dim,adj_mat,target_node,labelIn,error_label):

    label=np.copy(labelIn)
    #establish graph size
    num_nodes=np.shape(adj_mat)[0]
    #define blank error
    error=np.zeros(num_nodes)

    #add in z part of error
    label[target_node-1]=error_label[1]

    #add in x part of error
    for x in range(num_nodes):
        label[x]=((dim-1)*error_label[0]*adj_mat[target_node-1,x]+label[x])% dim

    return label

#***************************************************************************#
def qudit_through_channel(dim,numA,numB,adj_mat,target_node,coef_mat,param):

    input_coef_mat=np.copy(coef_mat)
    new_coef_mat=(1-param)*np.copy(coef_mat)

    for x in range(dim**2):
        error_label=list(product(np.arange(0,dim),repeat=2))[x]

        Alabels=list(product(np.arange(0,dim),repeat=numA))
        Blabels=list(product(np.arange(0,dim),repeat=numB))
        for row in range(dim**numA):
            for col in range(dim**numB):

                current_coef=input_coef_mat[row,col]

                if current_coef>0:
                    labelIn=np.array(Alabels[row]+Blabels[col])


                    #create list of effected labels
                    altered=[]

                    labelOut=label_update(dim,adj_mat,target_node,labelIn,error_label)

                    if compare_labels(labelOut,labelIn):
                        altered.append(labelOut)
                        #print('altered',altered)

                #determine which coefficients need to change and change them
                for entry in range(np.shape(altered)[0]):
                    focus=altered[entry]
                    for id_row in range(dim**numA):
                        for id_col in range(dim**numB):
                            label=np.array(Alabels[id_row]+Blabels[id_col])

                            if (not compare_labels(focus,label)):
                                new_coef_mat[id_row,id_col]+=current_coef*(param/(dim**2 -1))

    return new_coef_mat

#***************************************************************************#
def state_through_channel(dim,num_nodes,graph_type,param):

    numA,numB=assign_nodes(num_nodes,graph_type)

    if graph_type=='GHZ':
        adj_mat=get_GHZ_adj(num_nodes)
    elif graph_type=='line':
        adj_mat=get_lin_adj(num_nodes)

    coef_mat=perfect_coef_mat(dim,numA,numB)

    for x in range(1,num_nodes+1):
        coef_mat=qudit_through_channel(dim,numA,numB,adj_mat,x,coef_mat,param)

    return coef_mat

#***************************************************************************#
def save_depolarized_states(dim,num_nodes,graph_type,paramList):

    for x in range(np.shape(paramList)[0]):
        pstring=str(paramList[x])
        filename='Depolarized_Graph_States/{}_{}_{}_{}.txt'.format(dim,num_nodes,graph_type,pstring)
        if os.path.isfile(filename):
            continue
        else:
            afile=open(filename,'w')
            for row in state_through_channel(dim,num_nodes,graph_type,paramList[x]):
                np.savetxt(afile,row)

            afile.close()

    return
#***************************************************************************#
# save_depolarized_states(2,8,'GHZ',np.arange(0,0.7,0.01))
# print(state_through_channel(2,3,'GHZ',0.05)[1,0])
