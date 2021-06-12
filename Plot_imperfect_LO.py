import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text',usetex=True)

#dim=2
#Gate Errors#
#0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09

def markerFix(d):
    if d==2:
        mark='o'
    elif d==3:
        mark='s'
    elif d==5:
        mark='d'
    elif d==7:
        mark='X'
    elif d==11:
        mark='v'
    return mark

def dataFeed(dim,graph_type):
    if graph_type=='GHZ':
        if dim==2:
            #dim=2, GHZ, P2#
            ###########2     3    4    5    6    7    8    9
            GET=[None,None,0.05,0.04,0.04,0.03]#,0.03,0.02]#,0.11,0.1,0.09]
        elif dim==3:
            #dim=3, GHZ, P2#
            ###########2     3    4    5    6    7    8    9
            GET=[None,None,0.10,0.09,0.07,0.06]#,0.21,None,None]
        elif dim==5:
            ###########2     3    4    5    6    7    8    9
            GET=[None,None,0.16,0.15]
    elif graph_type=='line':
        if dim==2:
            ###############2     3    4    5    6    7    8    9
            GET=[None,0.055,0.063,0.055,0.065,0.062]#,None,0.062,None,0.06]#,0.25,0.26]
        elif dim==3:
            ###########2     3    4    5    6    7    8    9
            GET=[None,0.10,0.10,0.10,0.11,0.10]
        elif dim==5:
            ###########2     3    4    5    6    7    8    9
            GET=[None,0.15,0.16,0.14]#,0.11,0.10]
        elif dim==7:
            ###########2     3    4    5    6    7    8    9
            GET=[None,0.18,0.19,0.19]
        elif dim==11:
            ###########2     3    4    5    6    7    8    9
            GET=[None,0.24,0.25]

    return GET

def multiDim_plotter(dimlist,graph_type):
    plt.figure(figsize=(10,8))
    track=0
    cmap = plt.cm.get_cmap('plasma')
    inds=np.linspace(0,0.85,5)
    colors = [cmap(0),cmap(inds[1]),cmap(inds[2]),cmap(inds[3]),cmap(inds[4])]
    for dim in dimlist:
        GET=dataFeed(dim,graph_type)
        Nodes=np.arange(1,np.shape(GET)[0]+1)
        plt.plot(Nodes,GET,ls='-',c=colors[track],marker='o',label=r'$d={}$'.format(dim))
        # plt.scatter(np.arange(1,10),qcritD,label='Maximum parameter')
        #plt.scatter(np.arange(1,7),sFin,c=colourFix(dim),marker='x',label='d={}'.format(dim))
        track+=1
    plt.ylabel(r'Gate error threshold, $q_g$',fontsize=30 )
    plt.xlabel(r'Number of nodes, $N$',fontsize=30)
    if graph_type=='GHZ':
        plt.xticks(ticks=[3,4,5,6],fontsize=24)
    else:
        plt.xticks(ticks=[2,3,4,5,6],fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=24,loc=1)
    # plt.show()
    figname='../Figures/Good_Copies/Numeric_GE_{}'.format(graph_type)
    plt.savefig(figname,dpi=300,bbox_inches='tight')


def colourFix(N):
    if N==2:
        colour='black'
    elif N==3:
        colour='b'
    elif N==4:
        colour='navy'
    elif N==5:
        colour='mediumvioletred'
    elif N==6:
        colour='green'
    elif N==7:
        colour='m'
    elif N==8:
        colour='cyan'
    elif N==10:
        colour='red'
    return colour



# def dataFeed(num_nodes):
#
#     if num_nodes==2:
#         #Gate Errors#
#         ###########0   0.01 0.02 0.03 0.04 0.05 0.055 0.06 0.065 0.07 0.075 0.08 0.085 0.09
#         min_fids=[0.56,0.56,0.57,0.58,0.58,0.60,0.61,0.62,0.64,0.65,0.67,0.70,0.71,0.71,0.73,0.73]
#         max_fids=[1.00,0.99,0.98,0.97,0.96,0.94,0.93,0.91,0.89,0.86,0.83,0.80,0.76,0.76,0.75,0.74]
#         gate_er=[1-0,1-0.005,1-0.01,1-0.015,1-0.02,1-0.025,1-0.03,1-0.035,1-0.04,1-0.045,1-0.05,1-0.055,1-0.06,1-0.061,1-0.062,1-0.063]
#     elif num_nodes==3:
#         #Gate Errors#
#         ###########0   0.01 0.02 0.03 0.04 0.05 0.055 0.06 0.065 0.07 0.075 0.08 0.085 0.09
#         min_fids=[0.40,0.41,0.42,0.44,0.46,0.47,0.49,0.51,0.52,0.54,0.56,0.60,0.62]
#         max_fids=[1.00,0.99,0.98,0.97,0.95,0.93,0.92,0.90,0.87,0.84,0.80,0.76,0.71]
#         gate_er=[1-0,1-0.005,1-0.01,1-0.015,1-0.02,1-0.025,1-0.03,1-0.035,1-0.04,1-0.045,1-0.05,1-0.055,1-0.06]
#     elif num_nodes==4:
#         #Gate Errors#
#         ###########0   0.01 0.02 0.03 0.04 0.05 0.055 0.06 0.065 0.07 0.075 0.08 0.085 0.09
#         min_fids=[0.30,0.30,0.31,0.33,0.35,0.36,0.40,0.42,0.44,0.46,0.48,0.53,0.55,0.55,0.55]
#         max_fids=[1.00,0.99,0.97,0.95,0.93,0.90,0.88,0.84,0.81,0.77,0.72,0.67,0.61,0.60,0.59]
#         gate_er=[1-0,1-0.005,1-0.01,1-0.015,1-0.02,1-0.025,1-0.03,1-0.035,1-0.04,1-0.045,1-0.05,1-0.055,1-0.06,1-0.061,1-0.062]
#     elif num_nodes==5:
#         #Gate Errors#
#         ###########0   0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
#         min_fids=[0.17,0.19,0.23,0.28,0.35,0.41]
#         max_fids=[1.00,0.95,0.90,0.82,0.73,0.64]
#         gate_er=[1-0,1-0.01,1-0.02,1-0.03,1-0.04,1-0.045]
#     elif num_nodes==6:
#         #Gate Errors#
#         ###########0   0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
#         min_fids=[0.16,0.17,0.18,0.18,0.20,0.21,0.25,0.27,0.29,0.31,0.38,0.38,0.41,0.44,0.44]
#         max_fids=[1.0,0.98,0.95,0.92,0.89,0.86,0.82,0.78,0.73,0.68,0.62,0.55,0.48,0.47,0.45]
#         gate_er=[1-0,1-0.005,1-0.01,1-0.015,1-0.02,1-0.025,1-0.03,1-0.035,1-0.04,1-0.045, 1-0.05,1-0.055,1-0.06,1-0.061,1-0.062]
#     elif num_nodes==7:
#         #Gate Errors#
#         ###########0   0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
#         min_fids=[0.08,0.11,0.13,0.17,0.23]
#         max_fids=[1.00,0.93,0.87,0.81,0.65]
#         gate_er=[1-0,1-0.01,1-0.02,1-0.03,1-0.04]
#     elif num_nodes==8:
#         #Gate Errors#
#         ###########0   0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
#         min_fids=[0.08,0.09,0.10,0.12,0.13,0.14,0.16,0.17,0.19,0.21,0.27,0.33,0.36,0.30,0.36]
#         max_fids=[1.00,0.97,0.93,0.90,0.85,0.81,0.77,0.72,0.66,0.61,0.54,0.47,0.40,0.38,0.37]
#         gate_er=[1-0,1-0.005,1-0.01,1-0.015,1-0.02,1-0.025,1-0.03,1-0.035,1-0.04,1-0.045,1-0.05,1-0.055,1-0.06,1-0.061,1-0.062]
#     elif num_nodes==10:
#         #Gate Errors#
#         ###########0   0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
#         min_fids=[0.05,0.06,0.08,0.10,0.12,0.17,0.22]
#         max_fids=[1.00,0.91,0.82,0.72,0.61,0.48,0.34]
#         gate_er=[1-0,1-0.01,1-0.02,1-0.03,1-0.04,1-0.05,1-0.06]
#     return min_fids,max_fids,gate_er
#
# def plotter(max_num_nodes):
#     for N in range(2,max_num_nodes+1,2):
#         min_fids,max_fids,gate_er=dataFeed(N)
#         min_fids=np.flip(min_fids)
#         max_fids=np.flip(max_fids)
#         gate_er=np.flip(gate_er)
#         # gate_er=np.arange(0,(np.shape(min_fids)[0])/100,0.01)
#         # gate_er=np.arange(1-(np.shape(min_fids)[0])/100+0.01,1.00,0.01)
#         # print(np.shape(gate_er),np.shape(min_fids))
#         # for entry in gate_er:
#         #     gate_er[entry]=1-gate_er[entry]
#         plt.plot(gate_er,min_fids,ls='--',marker=markerFix(N),c=colourFix(N),label='N={}'.format(N))
#         plt.plot(gate_er,max_fids,ls='--',marker=markerFix(N),c=colourFix(N))
#     plt.legend(fontsize=14)
#     plt.xlabel('Gate error parameter, p',fontsize=18)
#     plt.ylabel('Fidelity, F', fontsize=18)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.show()


###Rough search data:
#d=2
# #Gate Errors#
# ###########0   0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
# min_fids=[0.51,0.53,0.57,0.61,0.68]
# max_fids=[1.00,0.98,0.94,0.90,0.85]
# gate_er=[1-0,1-0.01,1-0.02,1-0.03,1-0.04]

# num_nodes==3:
#     #Gate Errors#
#     ###########0   0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
#     min_fids=[0.32,0.34,0.38,0.44,0.52]
#     max_fids=[1.00,0.97,0.91,0.86,0.78]
#     gate_er=[1-0,1-0.01,1-0.02,1-0.03,1-0.04]

# num_nodes==4:
#     #Gate Errors#
#     ###########0   0.01 0.02 0.03 0.04 0.05 0.055 0.06 0.065 0.07 0.075 0.08 0.085 0.09
#     min_fids=[0.17,0.20,0.23,0.27,0.35,0.38,0.50,]
#     max_fids=[1.00,0.96,0.89,0.85,0.75,0.69,0.56]
#     gate_er=[1-0,1-0.01,1-0.02,1-0.03,1-0.04,1-0.045,1-0.05]

# num_nodes==6:
#     #Gate Errors#
#     ###########0   0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
#     min_fids=[0.10,0.12,0.15,0.18,0.25,0.30]
#     max_fids=[1.00,0.94,0.83,0.78,0.65,0.58]
#     gate_er=[1-0,1-0.01,1-0.02,1-0.03,1-0.04,1-0.045]

# num_nodes==8:
#     #Gate Errors#
#     ###########0   0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
#     min_fids=[0.05,0.07,0.08,0.12,0.17]
#     max_fids=[1.00,0.92,0.78,0.72,0.56]
#     gate_er=[1-0,1-0.01,1-0.02,1-0.03,1-0.04]

# plotter(10)
multiDim_plotter([2,3,5,7,11],'line')
