import time
import os.path
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from math import isclose
from itertools import product
from operator import add
from utils import assign_nodes, match_labB_to_indB, match_labA_to_indA
from weyl_covariant_channel import state_through_channel

#********VARIABLES********************************************************#
#num_nodes: number of nodes in graph
#dim: dimension of qudit
#graph_type: GHZ, line
#input_type: oneParam, DP
#param: x for one param states, // related to input fidelity
#iterations: Number of times to repeat purification protocol
#first_subP: 'P1' or 'P2' // Choice determines which purification is performed first
#alternation: True, Flase // If true, alternate between protocols P1 and P2
#doplot: True, False // If true, plot fidelity over rounds, as well as success probability
#**************************************************************************#

start_time=time.time()

#**************************************************************************#
#Calculates and saves to a file the coefficients of the werner type states
#**************************************************************************#
def save_oneParam_coefficients(num_nodes,dim,graph_type,param):
    numA,numB=assign_nodes(num_nodes,graph_type)

    coef_mat=np.zeros((dim**numA,dim**numB))
    coef_mat[0,0]=param+(1-param)/(dim**num_nodes)

    for x in range(0,dim**numA):
        for y in range(0,dim**numB):
            if x==0 and y==0:
                continue
            else:
                coef_mat[x,y]=(1-param)/(dim**num_nodes)

    filename='../oneParam_Graph_States/{}_{}_{}_{}.txt'.format(dim,num_nodes,graph_type,param)
    if not os.path.isfile(filename):
        afile=open(filename,'w')
        for row in coef_mat:
            np.savetxt(afile,row)

        afile.close()
    return

#**************************************************************************#
#For one parameter family analysis, determine minimum purification fidelity
#for a given graph type and state dimension
#**************************************************************************#
def oneParam_min_fidelity(dim,num_nodes,graph_type,subP):
    numA,numB=assign_nodes(num_nodes,graph_type)

    if subP=='P1':
        critX=(dim**num_nodes - 2*dim**numA + 1)/(1-dim**num_nodes - dim**numA + dim**(num_nodes+numA))
    elif subP=='P2':
        critX=(dim**num_nodes - 2*dim**numB + 1)/(1-dim**num_nodes - dim**numB + dim**(num_nodes+numB))

    return critX

#**************************************************************************#
#For a given graph, loads from file of generates the input coefficient
#matrix in the graph state basis
#**************************************************************************#
def get_input_coefficients(num_nodes,dim,graph_type,input_type,param):

    numA,numB=assign_nodes(num_nodes,graph_type)

    #set coefficients
    if input_type=='oneParam':

        filename='../oneParam_Graph_States/{}_{}_{}_{}.txt'.format(dim,num_nodes,graph_type,param)
        if os.path.isfile(filename):
            coef_mat=np.loadtxt(filename).reshape(dim**numA,dim**numB)
        else:
            print('generate')
            #rows associated with A labels
            #columns with B labels
            coef_mat=np.zeros((dim**numA,dim**numB))

            coef_mat[0,0]= param+(1-param)/(dim**num_nodes)

            for x in range(0,dim**numA):
                for y in range(0,dim**numB):
                    if x==0 and y==0:
                        continue
                    else:
                        coef_mat[x,y]=(1-param)/(dim**num_nodes)

    elif input_type=='DP':
        pstring=str(param)
        filename='../Depolarized_Graph_States/{}_{}_{}_{}.txt'.format(dim,num_nodes,graph_type,pstring)
        if os.path.isfile(filename):
            coef_mat=np.loadtxt(filename).reshape(dim**numA,dim**numB)
        else:
            coef_mat=state_through_channel(dim,num_nodes,graph_type,param)

    return coef_mat
#**************************************************************************#
#Runs sub-protocol P1 on a given input state, specified by cmat_in and the
#type params dim, num_nodes, graph_type
#**************************************************************************#
def P1_update_coefficients(num_nodes,dim,graph_type,cmat_in):
    dim_list=list(range(dim))

    numA,numB=assign_nodes(num_nodes,graph_type) #determines node bi-partition

    normK=0
    for x in range(0,dim**numA):
        for y in range(0,dim**numB):
            for z in range(0,y):
                normK+=cmat_in[x,y]*cmat_in[x,z]+cmat_in[x,z]*cmat_in[x,y]
            normK+=(cmat_in[x,y])**2


    #Application of measurement condition
    indices=list(product(dim_list,repeat=numB))
    coef_mat=np.zeros((dim**numA,dim**numB))
    for x in range(dim**numA):
        for y in range(dim**numB):
            controlB=np.array(indices[y])

            for col_muB in range(dim**numB):
                muB=np.array(indices[col_muB])
                nuB=np.add((dim-1)*muB,controlB) % dim
                #get col corresponding to nuB
                col_nuB=match_labB_to_indB(dim,nuB)
                #add to coefficient
                coef_mat[x,y]+=(cmat_in[x,col_muB]*cmat_in[x,col_nuB])/(normK)

    return normK,coef_mat

#**************************************************************************#
#Runs subprotocol P2 on a given input state
#**************************************************************************#
def P2_update_coefficients(num_nodes,dim,graph_type,cmat_in):
    dim_list=list(range(0,dim))

    numA,numB=assign_nodes(num_nodes,graph_type)

    normK=0
    for x in range(0,dim**numB):
        for y in range(0,dim**numA):
            for z in range(0,y):
                normK+=cmat_in[y,x]*cmat_in[z,x]+cmat_in[z,x]*cmat_in[y,x]
            normK+=(cmat_in[y,x])**2

    indices=list(product(dim_list,repeat=numA)) #all labels
    coef_mat=np.zeros((dim**numA,dim**numB)) #blank coef matrix
    for x in range(dim**numA):
        controlA=np.array(indices[x])
        for y in range(dim**numB):

            for row_muA in range(dim**numA):
                muA=np.array(indices[row_muA])
                nuA=np.add((dim-1)*muA,controlA) % dim
                #get row corresponding to nuA
                row_nuA=match_labA_to_indA(dim,nuA)
                #add to coefficient
                coef_mat[x,y]+=(cmat_in[row_muA,y]*cmat_in[row_nuA,y])/(normK)

    return normK,coef_mat

#**************************************************************************#
#Wrapper to handle plotting in run_purification if doplot is true
#**************************************************************************#
def single_plot(fids,psuccs,pcum_list,subP):
    xdat=range(0,np.size(fids))

    fig,ax1=plt.subplots()

    ax2=ax1.twinx()
    ax1.plot(xdat,fids,'blue')
    ax2.plot(xdat,pcum_list,'co',label='Cumulative Probability')
    ax2.plot(xdat,psuccs,'ro',label='Stage probability')
    ax1.set_xlabel('Purifications applied (beginning with {})'.format(subP),fontsize=18)
    ax1.set_ylabel('Fidelity',color='blue',fontsize=18)
    ax2.set_ylabel('Probaility of success',fontsize=18)
    ax2.legend(fontsize=14)
    ax1.tick_params(axis="x", labelsize=18)
    ax1.tick_params(axis="y", labelsize=18)
    ax2.tick_params(axis="y", labelsize=18) 
    plt.show()

#**************************************************************************#
#Handles purification mostly for Werner type states
#**************************************************************************#
def run_purification(num_nodes,dim,graph_type,input_type,param,iters,subP,alternation,doplot):
    csubP=subP
    fids=[]
    psucc_inst=[None]
    pcum_list=[None]
    pcum=1
    #prepare input P1_update_coefficients
    coef_mat=get_input_coefficients(num_nodes,dim,graph_type,input_type,param)
    fids.append(coef_mat[0,0])

    for x in range(iters):
        if csubP == 'P1':
            normK,coef_mat=P1_update_coefficients(num_nodes,dim,graph_type,coef_mat)
            if alternation==True:
                csubP='P2'

        elif csubP == 'P2':
            normK,coef_mat=P2_update_coefficients(num_nodes,dim,graph_type,coef_mat)
            if alternation==True:
                csubP='P1'

        fids.append(coef_mat[0,0])
        psucc_inst.append(normK)
        pcum*=normK
        pcum_list.append(pcum)

    if doplot == True:
        single_plot(fids,psucc_inst,pcum_list,subP)

    return fids

#**************************************************************************#
#Calls to run purification and plot for many dimensions, mostly for Werner
#input states
#**************************************************************************#
def manyD_plot(num_nodes,dimlist,graph_type,input_type,param,iters,subP,alternation):

    xdat=range(iters+1)
    for dim in dimlist:
        ydat=run_purification(num_nodes,dim,graph_type,input_type,param,iters,subP,alternation,False)
        plt.plot(xdat,ydat,'o',label='Dimension = {}'.format(dim))
    plt.legend()
    plt.locator_params(axis='x',nbins=iters+1)
    plt.xlabel('Number of purifications (beginning with {})'.format(subP))
    plt.ylabel('Fidelity')
    plt.show()
#***************************************************************************#
#for depolarized state
#***************************************************************************#
def get_fidsOut(param_tuple):
    """wrapper preparing for using map (which has issues with multiple variables
    -> pack all vars in one tuple). x[0]<-paramList, x[1]<-dim,x[2]<-num_nodes,x[3]=repeats
    x[4]<-graph_type, x[5]<-iters,x[6]<-FI==fidsIn proxy,x[7]<-csubP, x[8]<-alternation
    x[9]<-FO==fidsOut proxy,x[10]<-x"""

    paramList,dim,num_nodes,repeats,graph_type,iters,fidsIn,csubP,alternation,FO,x=param_tuple #unpack
    param=paramList[x]

    clean_coef_mat=get_input_coefficients(num_nodes,dim,graph_type,'DP',param)
    coef_mat=np.copy(clean_coef_mat)

    for z in range(iters):

        fidin=coef_mat[0,0]
        fidsIn[x+(z*repeats)]=coef_mat[0,0]

        if csubP=='P1':
            normK,coef_mat=P1_update_coefficients(num_nodes,dim,graph_type,coef_mat)
            if alternation==True:
                csubP='P2'

        elif csubP=='P2':
            normK,coef_mat=P2_update_coefficients(num_nodes,dim,graph_type,coef_mat)
            if alternation==True:
                csubP='P1'

        else:
            print('field subP should be specified as either P1 or P2 \n', 'Example: to start with P1 enter P1')
            return

        FO[x+(z*repeats)]=coef_mat[0,0]

    # return FO,FI


#**************************************************************************#
def run_depolarized_study(dim,num_nodes,graph_type,paramList,subP,iters,alternation,plotting):
    csubP=subP
    repeats=np.shape(paramList)[0]
    fidsOut=np.zeros((iters*repeats,))
    fidsIn=np.zeros((iters*repeats,))
    slopes=[None]

    #set up multiprocessing
    manager=multiprocessing.Manager() #create manager to handle shared objects
    FO=manager.Array('f',fidsOut) #Proxy for shared array
    FI=manager.Array('f',fidsIn) #Proxy for shared array
    mypool=multiprocessing.Pool() #Create pool of worker processes

    #update fidelities
    mypool.map(get_fidsOut,[(paramList,dim,num_nodes,repeats,graph_type,iters,FI,csubP,alternation,FO,x) for x in range(repeats)])
    #find critical points from each round of purification and calculate slopes
    for z in range(iters):
        for y in range(1,repeats-1):
            if (FO[(y-1) +(z*repeats)]>FI[y-1]) and (FO[(y+1)+(z*repeats)]<FI[y+1]):
                if z==(iters-1):
                    qcrit=paramList[y]
            slopes.append((FO[y + (z*repeats)]-FO[(y-1) + (z*repeats)])/(paramList[y]-paramList[y-1]))
        slopes.append((FO[repeats-1+(z*repeats)]-FO[(repeats-2+(z*repeats))])/(paramList[repeats-1]-paramList[repeats-2]))
        slopes.append(None)

    #Keep record critical slopes
    filename='../Limit_q/{}_{}_{}_qlim.txt'.format(dim,graph_type,subP)
    for z in range(iters):
        for y in range(1,repeats):
            if slopes[y+(z*repeats)]!=None and slopes[y+1+(z*repeats)]!=None and slopes[y-1+(z*repeats)]!=None:
                if abs(slopes[y+(z*repeats)])>abs(slopes[y-1+(z*repeats)]) and abs(slopes[y+(z*repeats)])>abs(slopes[y+1+(z*repeats)]) and abs(slopes[y+(z*repeats)])>1 and z>(iters/2):
                    q_val=paramList[y]
                    afile=open(filename,'a')
                    afile.write('Nodes {}, iteration {}, slope: {}, critical point at q value: {}\n'.format(num_nodes,z,slopes[y+(z*repeats)],q_val))
                    afile.close()
    try: qcrit
    except NameError: qcrit=None
    if qcrit!=None:
        #Keep record of qcrit
        filename='../Critical_q/{}_{}_{}_qcrit.txt'.format(dim,graph_type,subP)
        afile=open(filename,'a')
        afile.write('Nodes {}, qcrit : {}\n'.format(num_nodes,qcrit))
        afile.close()

    #deactivate plotting for larger states--for memory
    if num_nodes>5:
        plotting=False
    #plot
    if plotting==True:
        plt.figure()
        plt.plot(paramList,FI[0:repeats],label='Initial Fidelity')
        for z in range(iters):
            if (z%2)!=0 or z==0 or z==(iters-1):
                plt.plot(paramList,FO[z*repeats:((z+1)*repeats)],label='F out iteration {}'.format(z))
        plt.legend()
        plt.xlabel('Depolarization channel parameter q',fontsize=18)
        plt.ylabel('Fidelity to perfect graph state', fontsize=18)
        plt.title('{}, dim={}, N={}, Initial {}'.format(graph_type,dim,num_nodes,subP))
        figname='../Figures/DP_{}_{}_{}_{}.jpg'.format(dim,num_nodes,graph_type,subP)
        plt.savefig(figname,dpi=300)

        plt.figure()
        for z in range(iters):
            if (z%2)!=0 or z==0 or z==(iters-1):
                plt.plot(paramList,slopes[(z*repeats):((z+1)*repeats)],label='Iteration {}'.format(z))
        plt.legend()
        plt.xlabel('Depolarization channel parameter q', fontsize=18)
        plt.ylabel('Instantaneous rate of change of fidelity')
        plt.title('{}, dim={}, N={}, Initial {}'.format(graph_type,dim,num_nodes,subP))
        figname='../Figures/Slopes_DP_{}_{}_{}_{}.jpg'.format(dim,num_nodes,graph_type,subP)
        plt.savefig(figname,dpi=300)
        plt.close()


#**************************************************************************#
# print("--- %s seconds ---" % (time.time()-start_time))
