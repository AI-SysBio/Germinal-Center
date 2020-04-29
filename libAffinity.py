import numpy as np
from math import log, exp
import matplotlib.pyplot as plt
import sys
import random

from scipy.stats import binom
from collections import Counter


"""
  This library provides all the related functions to affinity and
  antigen/antibodies interactions
"""



class Cell():
    
    """
    Attributes
    ---------- 
    cMyc : double
        Current concentration of cMyc in the cell (division counter)
        
    affinity : double
        Affinity with the target antibody
        
    pMHC : float
        Concentration of the pMHC complex
        
    ancestor : int
        index of the seeder cell it originates from
        
    DNAseq : string
        IgV DNA string of the cell, used for affinity computation
        
    TChelp: float
        Cumulated Tfh signal
        
    tstart: float
        Time at which the cell entered in its last state
    """
       
    def __init__(self, cMyc = 0, DNAseq = [], ancestor = 0, pMHC = 0, TChelp = 0, tstart = 0):
        
        #important parameters
        self.cMyc = cMyc
        self.DNAseq = DNAseq
        self.affinity = affinity(DNAseq)
        self.pMHC = pMHC
        self.TChelp = TChelp
        self.tstart = tstart
        self.ancestor = ancestor


def affinity(seq, seq0 = None):
    
    """
       Compute the affinity to an antibody from the antigen DNA seq
       Norm. Hamming distance = 0 => affinity = 1
       Norm. Hamming distance = 1 => affinity = 0
    """
    Nsite = len(seq)
    if Nsite == 0:
        return 0
    
    if seq0 == None:
        seq0 = np.zeros(Nsite) # the target is set to a string full of 0
    H = Hamming_dist(seq,seq0)/Nsite
    aff = 1 - H
    
    return aff



def generate_BCR(Lseq,g_site, affinity0):
    
    LCB = np.random.randint(0,g_site,Lseq)
    H = int(affinity(LCB)*Lseq)
    while  H != int(affinity0*Lseq):
        
        if H < int(affinity0*Lseq):
            index_target = int(Lseq * (1-affinity(LCB)) * random.random())
            index = 0
            nchange = 0
            for n in range(Lseq):
                if LCB[n] != 0:
                    index += 1
                if index == index_target:
                    nchange = n
            LCB[nchange] = 0
            
        else:
            index_target = int(Lseq * affinity(LCB) * random.random())
            index = 0
            nchange = 0
            for n in range(Lseq):
                if LCB[n] == 0:
                    index += 1
                if index == index_target:
                    nchange = n
            LCB[nchange] = int(3*random.random() + 1)
            
                    
        H = int(affinity(LCB)*Lseq)

    if int(affinity(LCB)*Lseq) != int(affinity0*Lseq):
        LCB = generate_BCR(Lseq,g_site, affinity0)
        
    return LCB
    







def av_affinity(cell_list, which = [0,1]):
    
    aff = 0
    for i in which:
        if len(cell_list[i]) == 0:
            aff_i = 0
        else:
            aff_i = 0
            for j in range(len(cell_list[i])):
                aff_i += cell_list[i][j].affinity

        aff += aff_i
            
    tot = 0
    for i in which:
        tot += len(cell_list[i])
    
    if tot ==0:
        return 0
    else:
        aff /= tot
    
    return aff


def get_BCR(GCdata,n_seeder):
    """
    return the list of BCR sequence
    Input : GCdata = list containing all the cell in the current GC
    Output : list[ancestor][cell] containing the BCR sequence of centrocytes for each ancestors
    """
    
    GC_CB = GCdata[0] + GCdata[1] + GCdata[2] #only considering centroblasts and centrocyte
    BCRa = [[] for i in range(n_seeder)]
    for i in range(len(GC_CB)):
        a = GC_CB[i].ancestor
        BCRa[a].append(GC_CB[i].DNAseq)
      
    BCR = []
    for a in range(n_seeder):
        BCR.append(np.array(BCRa[a]))
        
    return BCR
    
    
def clonal_heterogeneity(BCR, timepoints):
    """
    return the clones evolution in GC
    Input : BCR = 4D list containing the BCR cel list of each ancestors at each time step BCR[t][a][cell][seq]
    Output : 2 array containing the number of clones and subclones at each time points
    """
    
    n_clones = np.zeros(timepoints+1) #number of ancestors re;aining
    NDS_true = np.zeros(timepoints+1) #normalized dominance score (to compare with exp data)
    NDS_10 = np.zeros(timepoints+1)
    for t in range(len(BCR)):
        
        anc_count = np.array([len(BCR[t][a]) for a in range(len(BCR[t]))])
        abest = np.argmax(anc_count)
        if anc_count[abest] > 0:
            NDS_true[t] = anc_count[abest] / np.sum(anc_count)
            n_clones[t] = np.size(np.where(anc_count>0))

            
        #compute a modified NDS to account for the fact that the experimental paper only has 10 colors
        anc_count_10 = np.zeros(10)
        for a in range(len(BCR[t])):
            anc_count_10[a%10] += anc_count[a]
        abest = np.argmax(anc_count_10)
        if anc_count_10[abest] > 0:
            NDS_10[t] = anc_count_10[abest] / np.sum(anc_count_10)
    
    return n_clones, NDS_true, NDS_10


def get_affinity(BCR):
    """
    return the BCR affinity of each cell from the current GCdata
    Input : BCR = 4D list containing the BCR cel list of each ancestors at each time step BCR[t][a][cell][seq]
    Output : 3Darray = BCR affinity [ncell] for each clone[n_clone init] for each time step[t]
    """
    
    aff = []
    for t in range(len(BCR)):
        aff_t = []
        for a in range(len(BCR[t])):
            if len(BCR[t][a]) == 0:
                aff_t.append(np.array([]))
            else:
                aff_t.append(np.array([affinity(BCR[t][a][i,:]) for i in range(len(BCR[t][a]))]))
        aff.append(aff_t)
  
    return aff




def Hamming_dist(A,B):
    """Returns the Hamming distance between array A and B"""
    return np.sum(A != B)


