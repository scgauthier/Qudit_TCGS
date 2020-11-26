import matplotlib.pyplot as plt
import numpy as np
from math import log

#function applies specifically to binary-like mixture initial states
#i.e. states where only qudits belonging to Va undergo dephasing
#function determines the updated fidelity after one round of protocol P1
def one_parameter_family(Fin,dim,subset_node_num):
    Fout=(Fin**2)/(Fin**2 + ((1-Fin)**2)/(dim**subset_node_num -1))
    return Fout

def one_param_fam_psucc(Fin,dim,subset_node_num):
    psucc=Fin**2 + ((1-Fin)**2)/(dim**subset_node_num - 1)
    return psucc

def one_param_fam_Von_Neuman_Ent(Fid,dim,subset_node_num):
    return (Fid*log((1-Fid)/(Fid*(dim**subset_node_num -1))) - log((1-Fid)/(dim**subset_node_num -1)))

def plot_one_param_fam_fout_vs_fin(subset_node_num,dimList,fidList):

    for y in range(0,np.size(dimList)):
        dim=dimList[y]
        fidOutList=[]
        for x in range(0,np.size(fidList)):
            fidOutList.append(one_parameter_family(fidList[x],dim,subset_node_num))
        plt.plot(fidList,fidOutList,label="dim = %s " % (dim))
    plt.legend()
    plt.xlabel('Input Fidelity')
    plt.ylabel('Output Fidelity')
    plt.show()

#print(one_parameter_family(0.501,3,1))

def plot_one_param_fam_psucc_vs_fin(subset_node_num,dimList,fidList):

    for y in range(0,np.size(dimList)):
        dim=dimList[y]
        psuccList=[]
        for x in range(0,np.size(fidList)):
            psuccList.append(one_param_fam_psucc(fidList[x],dim,subset_node_num))
        plt.plot(fidList,psuccList,label="dim = %s " % (dim))
    plt.legend()
    plt.xlabel('Input Fidelity')
    plt.ylabel('Protocol Probability of Success')
    plt.show()

def plot_one_param_fam_entin_vs_fid(subset_node_num,dimList):
    for y in range(0,np.size(dimList)):
        dim=dimList[y]
        # entDiff=[]
        ents=[]
        min_fid=1/(dim**subset_node_num)
        fidList=np.arange(min_fid+0.001,1,0.01)
        for x in range(0,np.size(fidList)):
            fid=fidList[x]
            ents.append(one_param_fam_Von_Neuman_Ent(fid,dim,subset_node_num))
            # fidOut=one_parameter_family(fid,dim,subset_node_num)
            # entOut=one_param_fam_Von_Neuman_Ent(fidOut,dim,subset_node_num)
            # entDiff.append(entIn-entOut)
        plt.plot(fidList,ents,label="d = %s " % (dim))
    plt.legend()
    plt.xlabel('Initial Fidelity')
    plt.ylabel('Initial Von Neuman Entropy')
    plt.show()

def plot_one_param_fam_entdiff_vs_entin(subset_node_num,dimList):
    for y in range(0,np.size(dimList)):
        dim=dimList[y]
        entDiff=[]
        entsIn=[]
        min_fid=1/(dim**subset_node_num)
        fidList=np.arange(min_fid+0.001,1,0.01)
        for x in range(0,np.size(fidList)):
            fid=fidList[x]
            entIn=one_param_fam_Von_Neuman_Ent(fid,dim,subset_node_num)
            entsIn.append(entIn)
            fidOut=one_parameter_family(fid,dim,subset_node_num)
            entOut=one_param_fam_Von_Neuman_Ent(fidOut,dim,subset_node_num)
            entDiff.append(entIn-entOut)
        plt.plot(entsIn,entDiff,label="d = %s " % (dim))
    plt.legend()
    plt.xlabel('Initial Von Neuman Entropy')
    plt.ylabel('Difference between Initial and Final Von Neuman Entropy')
    plt.show()


# plot_one_param_fam_entin_vs_fid(1,[2,3,5,7,11,13])
plot_one_param_fam_entdiff_vs_entin(1,[2,3,5,7,9,11,13])

#plot_one_param_fam_psucc_vs_fin(5,[2,3,5,7,11,13],np.arange(0.501,1,0.01))
