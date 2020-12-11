import time
import os.path
import numpy as np
from itertools import product
from weyl_covariant_channel import compare_labels, label_update, assign_nodes, perfect_coef_mat
from quditgraphstates import get_GHZ_adj

#***************************************************************************#
#Helper function to match a state label to a matrix index
#***************************************************************************#
def match_label_to_index(dim,numA,numB,label):

    labA=np.flip(label[0:numA])
    labB=np.flip(label[numA:])

    trkA=0
    trkB=0

    #Get A index
    for x in range(np.size(labA)):
        trkA+=labA[x]*(dim**x)

    for x in range(np.size(labB)):
        trkB+=labB[x]*(dim**x)

    return trkA,trkB

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
def perfect_coef_mat(dim,numA,numB):
    coef_mat=np.zeros((dim**numA,dim**numB))
    coef_mat[0,0]=1
    return coef_mat
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

#define the adjacency matrix of an N=num_nodes GHZ state
#**************************************************************************#
def get_GHZ_adj(num_nodes):
	adj_mat=np.zeros((num_nodes,num_nodes))
	for x in range(1,num_nodes):
		#ones in first row
		adj_mat[0,x]=1
		#ones in first column
		adj_mat[x,0]=1
	return adj_mat

#**************************************************************************#
def get_lin_adj(num_nodes):
	if (num_nodes % 2)!=0:
		numA=(num_nodes+1)/2
	else:
		numA=num_nodes/2
	numA=int(numA)

	adj_mat=np.zeros((num_nodes,num_nodes))
	for x in range(0,numA):
		if x==0:
			adj_mat[x,numA]=1
			adj_mat[numA,x]=1
		elif x==(numA-1):
			adj_mat[x,numA+x-1]=1
			adj_mat[numA+x-1,x]=1
		else:
			adj_mat[x,numA+x-1]=1
			adj_mat[x,numA+x]=1
			adj_mat[numA+x-1,x]=1
			adj_mat[numA+x,x]=1

	return adj_mat
