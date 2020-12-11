import os.path
import numpy as np
from itertools import product
import multiprocessing
from utils import compare_labels,label_update,assign_nodes,perfect_coef_mat,get_lin_adj,get_GHZ_adj

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
                    #graphs state basis label of coefficient to be updated
                    focus=altered[entry]
                    #get matrix indices of label
                    id_row,id_col=match_label_to_index(dim,numA,numB,focus)
                    #update coeficient matrix
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
        filename='../Depolarized_Graph_States/{}_{}_{}_{}.txt'.format(dim,num_nodes,graph_type,pstring)
        if os.path.isfile(filename):
            continue
        else:
            afile=open(filename,'w')
            for row in state_through_channel(dim,num_nodes,graph_type,paramList[x]):
                np.savetxt(afile,row)

            afile.close()

    return

#***************************************************************************#
