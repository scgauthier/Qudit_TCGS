import time
import os.path
import matplotlib.pyplot as plt
import numpy as np
from math import isclose
from itertools import product
from operator import add
from weyl_covariant_channel import assign_nodes, state_through_channel

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


    return normK,coef_mat

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

    indices=list(product(dim_list,repeat=numA))
    coef_mat=np.zeros((dim**numA,dim**numB))
    for control in range(0,dim**numA):
        indexControl=np.array(indices[control])
        # good_entries=[]
        for y in range(0,dim**numA):
            indexA1=np.array(indices[y])

            for z in range(0,y+1):
                indexA2=np.array(indices[z])

                for entry in range(0,numA):
                    if ((indexA1+indexA2+(dim-1)*indexControl) % dim)[numA-1-entry]!=0:
                        break
                    elif entry==numA-1 and y!=z:
                        for x in range(0,dim**numB):
                            coef_mat[control,x]+=(cmat_in[y,x]*cmat_in[z,x]+cmat_in[z,x]*cmat_in[y,x])/normK
                    elif entry==numA-1 and y==z:
                        for x in range(0,dim**numB):
                            coef_mat[control,x]+=((cmat_in[y,x])**2)/normK


    return normK,coef_mat

#**************************************************************************#
def single_plot(fids,psuccs,pcum_list,subP):
    xdat=range(0,np.size(fids))

    fig,ax1=plt.subplots()

    ax2=ax1.twinx()
    ax1.plot(xdat,fids,'blue')
    ax2.plot(xdat,pcum_list,'co',label='Cumulative Probability')
    ax2.plot(xdat,psuccs,'ro',label='Stage probability')
    ax1.set_xlabel('Purifications applied (beginning with {})'.format(subP))
    ax1.set_ylabel('Fidelity',color='blue')
    ax2.set_ylabel('Probaility of success')
    ax2.legend()
    plt.show()

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
def run_depolarized_study(dim,num_nodes,graph_type,paramList,subP,iters,alternation,plotting):
    csubP=subP
    delta_fid=np.zeros((iters,np.shape(paramList)[0]))
    fidsOut=np.zeros((iters,np.shape(paramList)[0]))
    fidsIn=np.zeros((iters,np.shape(paramList)[0]))

    for x in range(np.shape(paramList)[0]):
        param=paramList[x]
        coef_mat=get_input_coefficients(num_nodes,dim,graph_type,'DP',param)

        for z in range(iters):

            fidin=coef_mat[0,0]
            fidsIn[z,x]=coef_mat[0,0]

            if csubP=='P1':
                normK,coef_mat=P1_update_coefficients(num_nodes,dim,graph_type,coef_mat)
                delta_fid[z,x]=(coef_mat[0,0]-fidin)
                if alternation==True:
                    csubP='P2'

            elif csubP=='P2':
                normK,coef_mat=P2_update_coefficients(num_nodes,dim,graph_type,coef_mat)
                delta_fid[z,x]=(coef_mat[0,0]-fidin)
                if alternation==True:
                    csubP='P1'

            else:
                print('field subP should be specified as either P1 or P2 \n', 'Example: to start with P1 enter P1')
                return

            fidsOut[z,x]=coef_mat[0,0]

    for z in range(iters):
        for y in range(1,np.shape(paramList)[0]-1):
            if (fidsOut[z,y-1]>fidsIn[0,y-1]) and (fidsOut[z,y+1]<fidsIn[0,y+1]):
                if z==(iters-1):
                    qcrit=paramList[y]
            if (isclose(fidsOut[z,y],(1/dim),abs_tol=2e-3) and not isclose(fidsOut[z,y+1],(1/dim),abs_tol=4e-3)):
                filename='../Limit_q/{}_{}_{}_qlim.txt'.format(dim,graph_type,subP)
                afile=open(filename,'a')
                afile.write('Nodes {}, iteration {}, qlim : {}\n'.format(num_nodes,z,paramList[y]))
                afile.close()
            elif (isclose(fidsOut[z,y],(1/dim),abs_tol=2e-3) and not isclose(fidsOut[z,y-1],(1/dim),abs_tol=4e-3)):
                filename='../Limit_q/{}_{}_{}_qlim.txt'.format(dim,graph_type,subP)
                afile=open(filename,'a')
                afile.write('Nodes {}, iteration {}, qlim : {}\n'.format(num_nodes,z,paramList[y]))
                afile.close()

    #Keep record of qcrit
    filename='../Critical_q/{}_{}_{}_qcrit.txt'.format(dim,graph_type,subP)
    afile=open(filename,'a')
    afile.write('Nodes {}, qcrit : {}\n'.format(num_nodes,qcrit))
    afile.close()

    #plot
    if plotting==True:
        plt.figure()
        plt.plot(paramList,fidsIn[0,:],label='Initial Fidelity')
        for z in range(iters):
            plt.plot(paramList,fidsOut[z,:],label='F out iteration {}'.format(z))
        plt.legend()
        plt.xlabel('Depolarization channel parameter q',fontsize=18)
        plt.ylabel('Fidelity to perfect graph state', fontsize=18)
        plt.title('{}, dim={}, N={}, Initial {}'.format(graph_type,dim,num_nodes,subP))
        figname='../Figures/DP_{}_{}_{}_{}'.format(dim,num_nodes,graph_type,subP)
        plt.savefig(figname,dpi=300,format='jpg')

#**************************************************************************#


# run_depolarized_study(2,3,'GHZ',np.arange(0,0.6,0.01),'P1',10,True,True)

print("--- %s seconds ---" % (time.time()-start_time))
