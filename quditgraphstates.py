import numpy as np
import time
from math import sqrt,pi
from cmath import exp
from itertools import product

# start_time=time.time()
#Specify graph -- specify adjacency matrix of size n x n.
def specify_graph(sz):
	sz=int(sz)
	print('enter each row of the adjacency matrix entries in order')
	adjMat = [[int(input()) for x in range (sz)] for y in range (sz)]

	return adjMat

def getbasis(dim):
	Basis=[]
	#define basis vectors
	for val in range(0,dim):
		basisvec=np.zeros((dim,1))
		basisvec[(val,0)]=1
		Basis.append(basisvec)
	return Basis


#define a phase matrix for dimension d, raised to power pow
def phaseD(dim,pow):

	phase_mat=complex(1,0)*np.zeros((dim,dim))
	for val in range(0,dim):
		basisvec=np.zeros((dim,1))
		basisvec[(val,0)]=1
		projector=np.kron(basisvec,np.transpose(basisvec))

		phase_mat+=exp((2*pi*complex(0,1)*val)/dim)*projector

	#allow for powers of matrix
	if pow==dim:
		phase_mat=np.linalg.matrix_power(phase_mat,0)
	else:
		phase_mat=np.linalg.matrix_power(phase_mat,int(pow))

	return phase_mat

def shiftD(dim,pow):

	shift_mat=complex(1,0)*np.zeros((dim,dim))
	Basis=getbasis(dim)

	for val in range(0,dim):
		rais_op= np.kron(Basis[(val+1) % dim],np.transpose(Basis[val]))
		shift_mat+=rais_op
	#allow powers of operator
	if pow==dim:
		shift_mat=np.linalg.matrix_power(shift_mat,0)
	else:
		shift_mat=np.linalg.matrix_power(shift_mat,int(pow))
	return shift_mat

def inmodD(dim,pow):

	inmod_mat=complex(1,0)*np.zeros((dim,dim))
	Basis=getbasis(dim)

	for val in range(0,dim):
		inmod_mat[val,(dim-val)%dim]=1

	return inmod_mat



#define a kronecker product of shift and phase matrices applied
#according to vector inputs v and w
def XZvw(dim,v_string,w_string):
	#get node numbers from vector input
	num_nodes=np.size(v_string)
	#store terms to add to operator
	term_list=[]
	#decide on terms to add to operator
	#using entries of v_string, w_string
	for entry in range(0,num_nodes):
		#shift matrix to power v[entry]
		term = shiftD(dim,v_string[entry])
		#multiply by phase matrix to power w[entry] and store
		term=term.dot(phaseD(dim,w_string[entry]))
		#get rid of floating point errors
		for x in range(0,dim):
			for y in range(0,dim):
				#zero imaginary part
				if abs(term[x,y].imag)<1e-10:
					term[x,y]=complex(term[x,y].real,0)
				#zero real part
				elif abs(term[x,y].real)<1e-10:
					term[x,y]=complex(0,term[x,y].imag)
		#store term
		term_list.append(term)
	#turn term list into an operator
	for entry in range(0,num_nodes-1):
		if entry==0:
			operator=np.kron(term_list[0],term_list[1])
		else:
			operator=np.kron(operator,term_list[entry+1])
	return operator

#define a cphase between specified nodes
def CphaseAB(num_nodes,dim,nodeA,nodeB,pow):
	#blank complex matrix
	gate=complex(1,0)*np.zeros((dim**num_nodes,dim**num_nodes))
	#define basis vectors and build projector
	for val in range(0,dim):
		basisvec=np.zeros((dim,1))
		basisvec[(val,0)]=1
		control_proj= np.kron(basisvec,np.transpose(basisvec))
		#add matrices in order of node number to list
		termlist=[]
		for index in range(0,num_nodes):
			#nodeA is control
			if index==nodeA:
				term=control_proj
			#nodeB is target
			elif index==nodeB:
				term=phaseD(dim,val)
			#all other nodes identity
			else:
				term=np.identity(dim)
			termlist.append(term)
		#turn list of matrices into operator using kronecker product
		for index in range(0,num_nodes-1):
			if index==0:
				val_term=np.kron(termlist[0],termlist[1])
			else:
				val_term=np.kron(val_term,termlist[index+1])
		#total operator is linear combination
		gate+=val_term
	#allow powers of operators
	gate=np.linalg.matrix_power(gate,pow)
	#get rid of some floating point errors (if analytically=0)
	# tolerance=1e-10
	# for x in range(0,dim**num_nodes):
	# 	for y in range(0,dim**num_nodes):
	# 		if abs(gate[x,y].imag) < tolerance and abs(gate[x,y].real) < tolerance:
	# 			gate[x,y]=complex(0,0)
	# 		elif abs(gate[x,y].imag) < tolerance:
	# 			gate[x,y]=complex(gate[x,y].real,0)
	# 		elif abs(gate[x,y].real) < tolerance:
	# 			gate[x,y]=complex(0,gate[x,y].imag)

	return gate

def nonexpplicit_MPG(num_nodes,dim,Snode,direction,inStates):
	outStates=[]
	fullState=complex(1,0)*np.zeros((dim**(2*num_nodes),1))

	if direction!=1 and direction!=2:
		print('Direction should be specified as 1 for gate application in 12 direction or as 2 for application in 21 direction')
		return
	for x in range(0,np.shape(inStates)[0]):

		for y in range(0,dim):
			term_list1=[]
			term_list2=[]
			#get control for this loop
			diag=complex(1,0)*np.zeros((dim,dim))
			diag[y,y]=1

			for node in range(0,num_nodes):
				if node==Snode and direction==1:
					term_list1.append(diag)
					term_list2.append(shiftD(dim,y))
				elif node==Snode and direction==2:
					term_list1.append(shiftD(dim,y))
					term_list2.append(diag)
				else:
					term_list1.append(complex(1,0)*np.identity(dim))
					term_list2.append(complex(1,0)*np.identity(dim))

			val_term1=np.kron(term_list1[0],term_list1[1])
			val_term2=np.kron(term_list2[0],term_list2[1])
			for index in range(1,num_nodes-1):
				val_term1=np.kron(val_term1,term_list1[index+1])
				val_term2=np.kron(val_term2,term_list2[index+1])

			ket1=inStates[x][0]
			ket1=val_term1.dot(ket1)
			ket2=inStates[x][1]
			ket2=val_term2.dot(ket2)

			outStates.append([ket1,ket2])
			fullState+=np.kron(ket1,ket2)

	return outStates,fullState


#Multi-party gate: on node A with source state 1 and target state 2 for Direction
#1, vice verse for direction 2
def MPG(num_nodes,dim,Snode,direction):
	gate=complex(1,0)*np.zeros((dim**(2*num_nodes),dim**(2*num_nodes)))
	#Gate direction 12
	if direction==1:
		for x in range(0,dim):
			term_list=[]
			#get control for loop
			diag=complex(1,0)*np.zeros((dim,dim))
			diag[x,x]=1
			#Generate G-CNOT between nodes A and A' with A as source
			for node in range(0,2*num_nodes):
				#assess if node is current target
				if node==Snode:
					term=diag
				#target node in non-zero state
				elif node==(Snode+num_nodes):
					term=shiftD(dim,x)
				else:
					term=complex(1,0)*np.identity(dim)

				term_list.append(term)

			val_term=np.kron(term_list[0],term_list[1])
			for index in range(1,2*num_nodes-1):
				val_term=np.kron(val_term,term_list[index+1])

			gate+=val_term
	#Gate direction 21
	elif direction==2:
		for x in range(0,dim):
			term_list=[]
			#get control for loop
			diag=complex(1,0)*np.zeros((dim,dim))
			diag[x,x]=1
			#Generate G-CNOT between nodes A and A' with A as source
			for node in range(0,2*num_nodes):
				#assess if node is current target
				if node==Snode:
					term=shiftD(dim,x)
				#target node in non-zero state
				elif node==(Snode+num_nodes):
					term=diag
				else:
					term=complex(1,0)*np.identity(dim)

				term_list.append(term)

			val_term=np.kron(term_list[0],term_list[1])
			for index in range(1,2*num_nodes-1):
				val_term=np.kron(val_term,term_list[index+1])

			gate+=val_term
	else:
		print('Direction should be specified as 1 for gate application in 12 direction or as 2 for application in 21 direction')
		return

	# tolerance=1e-10
	# for x in range(0,dim**(2*num_nodes)):
	# 	for y in range(0,dim**(2*num_nodes)):
	# 		if abs(gate[x,y].imag) < tolerance and abs(gate[x,y].real) < tolerance:
	# 			gate[x,y]=complex(0,0)
	# 		elif abs(gate[x,y].imag) < tolerance:
	# 			gate[x,y]=complex(gate[x,y].real,0)
	# 		elif abs(gate[x,y].real) < tolerance:
	# 			gate[x,y]=complex(0,gate[x,y].imag)
	#
	return gate


#Modified MPCnot: on node A with source state 1 and taget state 2 for Direction
#1, vice verse for direction 2. Applies superposition of X^1 through X^d-1
#instead of X on target
def Alt_MPG(num_nodes,dim,Snode,direction):

	gate=complex(1,0)*np.zeros((dim**(2*num_nodes),dim**(2*num_nodes)))
	#Gate direction 12
	if direction==1:
		for x in range(0,dim):
			term_list=[]
			#get control for loop
			diag=complex(1,0)*np.zeros((dim,dim))
			diag[x,x]=1
			#Generate G-CNOT between nodes A and A' with A as source
			for node in range(0,2*num_nodes):
				#assess if node is current target
				if node==Snode:
					term=diag
				#target node in non-zero state
				elif node==(Snode+num_nodes):
					term=shiftD(dim,(x*(dim-1))%dim)
				else:
					term=complex(1,0)*np.identity(dim)

				term_list.append(term)

			val_term=np.kron(term_list[0],term_list[1])
			for index in range(1,2*num_nodes-1):
				val_term=np.kron(val_term,term_list[index+1])

			gate+=val_term
	#Gate direction 21
	elif direction==2:
		for x in range(0,dim):
			term_list=[]
			#get control for loop
			diag=complex(1,0)*np.zeros((dim,dim))
			diag[x,x]=1
			#Generate G-CNOT between nodes A and A' with A as source
			for node in range(0,2*num_nodes):
				#assess if node is current target
				if node==Snode:
					term=shiftD(dim,(x*(dim-1))%dim)
				#target node in non-zero state
				elif node==(Snode+num_nodes):
					term=diag
				else:
					term=complex(1,0)*np.identity(dim)

				term_list.append(term)

			val_term=np.kron(term_list[0],term_list[1])
			for index in range(1,2*num_nodes-1):
				val_term=np.kron(val_term,term_list[index+1])

			gate+=val_term
	else:
		print('Direction should be specified as 1 for gate application in 12 direction or as 2 for application in 21 direction')
		return

	# tolerance=1e-10
	# for x in range(0,dim**(2*num_nodes)):
	# 	for y in range(0,dim**(2*num_nodes)):
	# 		if abs(gate[x,y].imag) < tolerance and abs(gate[x,y].real) < tolerance:
	# 			gate[x,y]=complex(0,0)
	# 		elif abs(gate[x,y].imag) < tolerance:
	# 			gate[x,y]=complex(gate[x,y].real,0)
	# 		elif abs(gate[x,y].real) < tolerance:
	# 			gate[x,y]=complex(0,gate[x,y].imag)

	return gate

#prepare a graph state by specifying the number of nodes, the dimension and
#an adjacency matrix
def prepare_graph(dim,adjMat):

	#get num_nodes from adjMat
	num_nodes=np.shape(adjMat)[0]
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
			graph_state=CphaseAB(num_nodes,dim,a,b,int(adjMat[a,b])).dot(graph_state)
	#eliminate some floating point errors
	for entry in range(0,np.size(graph_state)):
		if abs(graph_state[entry].imag) < 1e-10:
			graph_state[entry]=complex(graph_state[entry].real,0)
	return graph_state

#define the adjacency matrix of an N=num_nodes GHZ state
def get_GHZ_adj(num_nodes):
	adj_mat=np.zeros((num_nodes,num_nodes))
	for x in range(1,num_nodes):
		#ones in first row
		adj_mat[0,x]=1
		#ones in first column
		adj_mat[x,0]=1
	return adj_mat

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


#Have function set the graph state construction as index vec[0]
#compare input state to all graph basis states for graph type
#return index of input state
def decide_index(dim,adjMat,unk_state):
	num_nodes=np.shape(adjMat)[0]
	#get the 0 index state for comparison
	graph_state=prepare_graph(dim,adjMat)
	#generate all strings of length num_nodes
	#with entries in finite field dimension d
	dim_list=[]
	for x in range(0,dim):
		dim_list.append(x)
	#get appropriately sized 0 vec for x entries
	zero_vec=np.zeros((num_nodes))
	#get trial states by systematically applying all
	#combinations of phase matrices
	for y in range(0,dim**num_nodes):
		current_string=list(product(dim_list,repeat=num_nodes))[y]
		trial_state=XZvw(dim,zero_vec,current_string).dot(graph_state)
		#compare trial and unknown state
		ratio=unk_state/trial_state

		state_shape=np.shape(unk_state)
		global_phase=ratio[0,0]
		#if they are the same, break and return index string
		if np.allclose(ratio/global_phase,np.ones(state_shape),atol=1e-5):
			print('match_found')
			return current_string, global_phase
		elif y==(dim**num_nodes - 1):
			print('no match found')

def decide_double_index(dim,adjMat,unk_state):

	num_nodes=np.shape(adjMat)[0]
	#get tensor product of two 0 index states for comparison base
	graph_state=prepare_graph(dim,adjMat)
	graph_state=np.kron(graph_state,graph_state)

	#generate all strings of length 2*num_nodes with entries
	#in finite field of order d
	dim_list=[]
	for x in range(0,dim):
		dim_list.append(x)
	#get appropriately sized 0 vec for X entries
	zero_vec=np.zeros((2*num_nodes))
	#get trial states by systematically applying all combinations of
	#phase matrices
	for y in range(0,dim**(2*num_nodes)):
		current_string=list(product(dim_list,repeat=2*num_nodes))[y]
		trial_state=XZvw(dim,zero_vec,current_string).dot(graph_state)
		#compare trial and unknown states
		ratio = unk_state/trial_state

		state_shape=np.shape(unk_state)
		global_phase=ratio[0,0]
		#if they are the same, break and return index string
		if np.allclose(ratio/global_phase,np.ones(state_shape),atol=1e-5):
			print('match found')
			return current_string, global_phase
		elif y==(dim**(2*num_nodes)-1):
			print('no match found')

#Next: write a function which takes in the label of a state,
#performs a specified operation, and returns label of new state
def update_label(dim,adjMat,label,operation):
	num_nodes=np.shape(adjMat)[0]
	#get the 0 index state for comparison
	graph_state=prepare_graph(dim,adjMat)
	#get zero vec of right size
	zero_vec=np.zeros((num_nodes))
	#prepare input state
	in_state=XZvw(dim,zero_vec,label).dot(graph_state)
	#do operation on input state and update label
	return decide_index(dim,adjMat,operation.dot(in_state))

#Square_cluster_ten=np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0,0],[1,0,1,0,0,0,1,0,0,0],[0,1,0,1,0,0,0,1,0,0],[0,0,1,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,0,0,1],[1,0,0,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,1,0,0],[0,0,1,0,0,0,1,0,1,0],[0,0,0,1,0,0,0,1,0,1],[0,0,0,0,1,0,0,0,1,0]])
#print(list(product([0,1,2],repeat=2)))
# graph_state=prepare_graph(3,get_GHZ_adj(5))
# update=XZvw(3,[0,0,0,0,2],[0,0,0,0,1]).dot(graph_state)
# print(decide_index(3,get_GHZ_adj(5),update))
#graph_state=prepare_graph(3,get_lin_adj(5))
#input=np.kron(XZvw(3,[0,0,0,0,0],[0,0,0,0,0]).dot(graph_state),XZvw(3,[0,0,0,0,0],[0,0,0,0,1]).dot(graph_state))
#update=XZvw(3,[0,0,0,0,0],[1,0,2,1,0]).dot(graph_state)
#print(decide_index(3,get_GHZ_adj(3),XZvw(3,[0,0,1,0,0],[1,0,0,0,0]).dot(exp(4*pi*complex(0,1)/3)*update)))
#print(np.identity(5)+exp(4*pi*complex(0,1)/5)*phaseD(5,1)+exp(8*pi*complex(0,1)/5)*phaseD(5,2)+exp(2*pi*complex(0,1)/5)*phaseD(5,3)+exp(6*pi*complex(0,1)/5)*phaseD(5,4))
#update=Alt_MPCnot(2,3,0,2).dot(Alt_MPCnot(2,3,0,2).dot(np.kron(graph_state,graph_state)))
#update=Alt_MPCnot(2,3,1,1).dot(Alt_MPCnot(2,3,1,1).dot(update))
#update=MPG(3,3,0,2).dot(np.kron(graph_state,graph_state))
#print(decide_double_index(3,get_GHZ_adj(2),update))
#print(update - np.kron(graph_state,graph_state))


#print(gate.dot(np.kron(Basis[2],Basis[0])))
#update=MPG(5,3,0,2).dot(input)
#update=MPG(5,3,1,2).dot(update)
#update=MPG(5,3,2,2).dot(update)
#update=Alt_MPG(3,3,2,2).dot(update)
#print(decide_double_index(3,get_GHZ_adj(3),update))
#print(update-input)

#outStates,fullState=nonexpplicit_MPG(7,3,0,1,[[graph_state,graph_state]])
#print(MPG(3,3,0,1).dot(np.kron(graph_state,graph_state))-fullState)
#print(fullState)



# print("--- %s seconds ---" % (time.time()-start_time))
