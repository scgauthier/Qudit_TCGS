import numpy as np
from math import sqrt,pi
from cmath import exp

#Specify graph -- specify adjacency matrix of size n x n.
def specify_graph(sz):
	sz=int(sz)
	print('enter each row of the adjacency matrix entries in order')
	adjMat = [[int(input()) for x in range (sz)] for y in range (sz)]

	return adjMat

#define a phase matrix for dimension d, raised to power pow
def phaseD(dim,pow):

	phase_mat=exp(2*pi*complex(0,1))*np.zeros((dim,dim))
	for val in range(0,dim):
		basisvec=np.zeros((dim,1))
		basisvec[(val,0)]=1
		projector=np.kron(basisvec,np.transpose(basisvec))

		phase_mat+=exp((2*pi*complex(0,1)*val)/dim)*projector

	if pow==dim:
		phase_mat=np.linalg.matrix_power(phase_mat,0)
	else:
		phase_mat=np.linalg.matrix_power(phase_mat,pow)

	return phase_mat
		

#define a cphase between specified nodes
def CphaseAB(num_nodes,dim,nodeA,nodeB):

	gate=exp(2*pi*complex(0,1))*np.zeros((dim**num_nodes,dim**num_nodes))
	for val in range(0,dim):
		basisvec=np.zeros((dim,1))
		basisvec[(val,0)]=1
		control_proj= np.kron(basisvec,np.transpose(basisvec))

		termlist=[]

	
		for index in range(0,num_nodes):
			if index==nodeA:
				term=control_proj

			elif index==nodeB:
				term=phaseD(dim,val)

			else:
				term=np.identity(dim)

			termlist.append(term)

		for index in range(0,num_nodes-1):
			if index==0:
				val_term=np.kron(termlist[0],termlist[1])
			else:
				val_term=np.kron(val_term,termlist[index+1])

		gate+=val_term

	return gate
	


#prepare a graph state by specifying the number of nodes, the dimension and 
#an adjacency matrix
def prepare_graph(num_nodes,dim,adjMat):

	#prepare a single d-dim plus state
	xplus=0
	for val in range(0,dim):
		basisvec=np.zeros((dim,1))
		basisvec[(val,0)]=1
		xplus+=(1/sqrt(dim)) * basisvec

	#prepare length N tensor product of plus states
	graph_state=xplus

	for val in range(0,num_nodes-1):
		graph_state=np.kron(graph_state,xplus)
		

	#apply Cphase gates according to adjacency matrix
	for a in range(0,num_nodes):
		for b in range(0,a):
			graph_state=CphaseAB(num_nodes,dim,a,b).dot(graph_state)

	for entry in range(0,np.size(graph_state)):
		if abs(graph_state[entry].imag) < 1e-10:
			graph_state[entry]=complex(graph_state[entry].real,0)

	return graph_state
	
def get_GHZ_adj(num_nodes):

	adj_mat=np.zeros((num_nodes,num_nodes))
	
	for x in range(1,num_nodes):

		#ones in first row
		adj_mat[0,x]=1
		#ones in first column
		adj_mat[x,0]=1

	return adj_mat
	
	



GHZ_five=np.array([[0,1,1,1,1],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]])
GHZ_four=np.array([[0,1,1,1,1],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]])

Square_cluster_ten=np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0,0],[1,0,1,0,0,0,1,0,0,0],[0,1,0,1,0,0,0,1,0,0],[0,0,1,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,0,0,1],[1,0,0,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,1,0,0],[0,0,1,0,0,0,1,0,1,0],[0,0,0,1,0,0,0,1,0,1],[0,0,0,0,1,0,0,0,1,0]])
graph_state=prepare_graph(5,3,GHZ_five)
#print(graph_state)

graph_state=prepare_graph(3,2,get_GHZ_adj(3))

print(get_GHZ_adj(3))






