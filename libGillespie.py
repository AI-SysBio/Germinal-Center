
"""
  This library provides all the necessary function for the gillepsie algorithm
  The particles has individual properties so has to be considered individually
    
  In the simulation, 5 types of reactants are considered:
      - Centroblast (Dark Zone) = CB
      - Centrocytes (Light Zone) = CC
      - Bounded Centrocytes (Light Zone) = CCTC
      - Free T follicular helper (Light Zone) = Tfh
      - Selected Centrocytes (CCsel)
      
        (Plus 3 additional cell types, leaving the GC)
      - Memory cells (Outside GC) = MC
      - Plasma cells (Outside GC) = PC
      - Dead cells (in heaven) = 0
      
   9 reactions are considered:
      - Centrocyte apoptosis:        CC -> 0
      - Centroblast migration:       CB -> CC
      - Centrocite unbinding:        [CCTC] -> CC + TC
      - Centrocyte recirculation:    CC -> CB
      - Centrocyte MC exit:          CC -> MC
      - Centrocyte PC exit:          CC -> PC (+ CB)
      - Centrocyte Tfh binding:      CC + TC = [CCTC]
      - Tfh switch:                  [CC1TC] + CC2 -> CC1 + [CC2TC]
      - Centroblast division:        CB -> 2CB
      
      - FDCantigen uptake:         CC -> CCr
      - refractoratory state exit: CCr -> CC (is it that important ?, yes it make harder to get antigen)
"""


import numpy as np
from math import log, exp, sqrt, gamma
import random
import timeit
import copy
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from libAffinity import Cell, affinity, av_affinity, get_BCR, generate_BCR, clonal_heterogeneity

"""
    Implementation details:

    The indexes of CB are stored as a sorted map to make the draw of Bcell division
    faster to implement

"""
        
    
    
        
def launch_Gillespie(GCdata_init, Tend, rates, param_intra, timepoints, Tfh_percentil):
      
    """
    Run the Gillespie simulation once, return the populations of each reactant at different timestep
    
    Input:
        N_GC_init   = 1D array,     number of initial cell for each reactants
        Tend        = float,        time length of the simulation
        rates       = 1D array,     rates of each reaction
        timepoints  = int,          number of timepoints to save
        
    Output:
        population  = 3D array,     population of each reactant at each time step
    """

 
    #Run simulation until final time is reached:
    t = 0
    ti = 0
    population = np.zeros((timepoints+1,len(GCdata_init)-1))
    affinity_ = np.zeros(timepoints+1)
    GCdata = copy.deepcopy(GCdata_init)
    n_seeder = np.max([GCdata[-1][i].ancestor for i in range(len(GCdata[-1]))])+1
    rhoTC = rates[-1]
    BCR = []
    n_Thelp = []
    
    stop_seeding = False
    
    
    
    while t < Tend: #Monte Carlo step
        
        
        set_Tcell(GCdata, rhoTC,t)
        
        #1 ----- compute propensities
        if t <= 3.5*24: #germinal center is empty until day 3.5
            t += Tend/timepoints/1.2
        
          
        else:
            propensity, stop_seeding = compute_propensity(GCdata, rates, t, stop_seeding)
            a0 = np.sum(propensity)
            
            if a0 == 0: #if propensities are zero, quicly end the simulation
                t += Tend/timepoints/1.2
                
            elif len(GCdata[0]) + len(GCdata[1]) > 2500: #if number of cells is too high, quickly end the simulation
                t += Tend/timepoints/1.2
                
            elif len(GCdata[0]) + len(GCdata[1]) > 2000 and t>14*24: #if number of cells is too high, quickly end the simulation
                t += Tend/timepoints/1.2
                 
            else:
                #2 ----- Generate random time step (exponential distribution)
                r1 = random.random()
                tau = 1/a0*log(1/r1)
                t += tau
                
                
                #3 ----- Chose the reaction mu that will occurs (depends on propensities)
                r2 = random.random()
                mu = 0
                p_sum = 0.0
                while p_sum < r2*a0:
                    p_sum += propensity[mu]
                    mu += 1
                mu = mu - 1 
                
            
                #4 ----- Perform the reaction
                perform_reaction(mu,GCdata,rates,param_intra, t, n_Thelp, Tfh_percentil, affinity_)
                #print(NCB0)
    
    
        if t >= ti*Tend/timepoints:
            starttemp = timeit.default_timer()
            
            GC_CB,GC_CC,GC_CCsel,GC_CCTC,GC_TC,GC_MC,GC_PC,GC_DCC,GC_DCB,Seeder_list = GCdata
            
            #if t>= 8*24:
            BCR.append(get_BCR(GCdata,n_seeder))
            population[ti,:] = np.array([len(GC_CB),len(GC_CC),len(GC_CCsel),len(GC_CCTC),len(GC_TC),len(GC_MC),len(GC_PC),len(GC_DCC),len(GC_DCB)])
            
            
            affinity_[ti] = av_affinity(GCdata, which = [1])

            ti += 1
            
    n_clones, _, NDS = clonal_heterogeneity(BCR, timepoints)
    properties = (np.concatenate((affinity_.reshape((-1,1)),n_clones.reshape((-1,1)),NDS.reshape((-1,1))),axis = 1))
    return population, properties, BCR, np.array(n_Thelp)
        
        
        
def initialize_GC(initial_parameters,param_intra):
    
    """
    Initialize GCdata with a dynamic list of cells
    
    Input:
        N_CB, N_CC, N_CCsel, N_CCTC, N_CC, N_TC, N_MC, N_PC, N_DC, N_DB = N_GC
    
    Output:
        list of list of Cell GCdata:
        - GCdata[0] = contains CB (Centroblast)
        - GCdata[1] = contains CC (Centrocyte)
        - GCdata[2] = contains CCsel (Selected Centrocyte)
        - GCdata[3] = contains CCTC
        - GCdata[4] = contains TC (T Follicular Helper)
        - GCdata[5] = contains MC (Memory Cells)
        - GCdata[6] = contains PC (Plasma Cells)
        - GCdata[7] = contains DCC (Dead Centrocyte)
        - GCdata[8] = contains DCB (Dead Centroblast)
        - GCdata[9] = contains the seeder list (before they enter the GC)
    """
    
    N_GC_init,n_seeder = initial_parameters
    Lseq,g_site,_,_,_,_ = param_intra 
    
    #Fill the germinal center with N_init cells
    N_init = np.sum([N_GC_init[i] for i in range(len(N_GC_init))])-N_GC_init[3] 
    LCB_chosen = np.zeros((N_init, Lseq))
    for i in range(N_init):
        LCB_chosen[i,:] = generate_BCR(Lseq,g_site, 0.4)

        
    #   (index i = 3 corresponds to Tcells)
    GCdata = []
    ci = 0
    for i in range(len(N_GC_init)):
        NGCi = []
        for j in range(N_GC_init[i]):
            if i != 3: #i=3 is Tcells, they dont have BCR
                DNAseq = LCB_chosen[ci]
                NGCi.append(Cell(cMyc = 0, DNAseq = DNAseq, ancestor = j))
                ci += 1
            else:
                NGCi.append(Cell(cMyc = 0, DNAseq = np.zeros(Lseq)))
        GCdata.append(NGCi)
        
    seederBCR = np.array([GCdata[-1][i].DNAseq for i in range(N_GC_init[-1])])
    
    return GCdata, seederBCR
        



def compute_propensity(GCdata, rates, t, stop_seeding):
    
    """
    Compute the propensity corresponding to each reaction
    the propensity is the reaction rate of a given reaction

      - Centroblast division:           CB -> 2CB 
      - Centroblast migration:          CB -> CC
      - Centrocyte antigen uptake       CC -> CC
      - Centrocyte Tfh binding:         CC + TC -> [CCTC]
      - Centrocyte unbinding:           CCTC -> CC + TC
      - Tfh switch:                     [CC1TC] + CC2 -> CC1 + [CC2TC]
      - Centrocyte fate:                CC -> 0 or CB or MC or PC
    """
    
    r_activation, r_division, r_migration, r_FDCencounter, r_TCencounter, r_unbinding, r_apoptosis, r_recirculate, r_exit, rhoTC = rates
    GC_CB,GC_CC,GC_CCsel,GC_CCTC,GC_TC,GC_MC,GC_PC,GC_DCC,GC_DCB,_ = GCdata
    NFDC = 250  #is constant through the whole simulation
    
    propensity = []
    propensity.append(len(GC_CB)   * r_division)                    #CB -> 2CB
    propensity.append(len(GC_CB)   * r_migration)                   #CB -> CC
    propensity.append((NFDC   * r_FDCencounter) * (len(GC_CC)>0) )             #CC -> CC (antigen uptake)
    propensity.append(len(GC_TC)   * r_TCencounter * (len(GC_CC)>0))    #CC + TC = [CCTC]
    propensity.append(len(GC_CCTC) * r_TCencounter * (len(GC_CC)>0))    #[CC1TC] + CC2 -> CC1 + [CC2TC]
    propensity.append(len(GC_CCTC) * r_unbinding)                   #CCTC -> CC + TC
    propensity.append(len(GC_CC) * r_apoptosis)
    propensity.append(len(GC_CCsel) * r_recirculate)
    propensity.append(len(GC_CCsel) * r_exit)
    
    
    if (len(GC_CB) + len(GC_CC) <= 1800)  and stop_seeding == False:
        propensity.append(r_activation)  
    else:
        stop_seeding = True
        propensity.append(0)  
     
    
    return np.array(propensity), stop_seeding
                
    

        
def perform_reaction(mu,GCdata,rates,param_intra, t, n_Thelp, Tfh_percentil, affinity_GC):
    r_activation, r_division, r_migration, r_FDCencounter, r_TCencounter, r_unbinding, r_apoptosis, r_recirculate, r_exit, rhoTC = rates
    Lseq,g_site,pshm,antigenthreshold, s, delta = param_intra 
    
    """
    Perform the reaction mu and modify the GC data accordingly
    
    Note about complexity:
        
        - append() operation is O(1)
        - When pop() is called from the end, the operation is O(1), while calling pop() from anywhere else is O(n)
          due to memory realocation. One trick to gain speed is to exchange values of the element you want to delete 
          with the last element, and then use pop().
          
    Reactions 5 is 30 times longer than the others (1 mu_s vs 100 mu_s)
    due to the SHM computation
    """
    
    GC_CB,GC_CC,GC_CCsel,GC_CCTC,GC_TC,GC_MC,GC_PC,GC_DCC,GC_DCB,Seeder_list = GCdata
    
    
    if mu == 9: #Bcell activation:    NB -> CB
        if len(Seeder_list) > 0:
            index = int(len(Seeder_list) * random.random())
            CellB = popcell(Seeder_list,index)
            GC_CB.append(CellB)
            GC_CB[-1].cMyc = 6
    
    
    elif mu == 0: #Centroblast division:       CB -> 2CB 
        
        index = int(len(GC_CB) * random.random())
        ndiv = GC_CB[index].cMyc
        rdiv_true = r_division * (ndiv != 0)
        
        if rdiv_true >= r_division*random.random():
            
            DNAseq_new1, nmut1, lethal1 = SHM_mutation(GC_CB[index].DNAseq,pshm,delta,g_site,s)
            DNAseq_new2, nmut2, lethal2 = SHM_mutation(GC_CB[index].DNAseq,pshm,delta,g_site,s)
                
            if lethal1 and lethal2: #both daughter cell die
                CellB = popcell(GC_CB,index)
                GC_DCB.append(CellB)
                GC_DCB.append(CellB)
                     
            elif lethal1 and not lethal2: #one cell survive
                GC_DCB.append(GC_CB[index])
                GC_CB[index].DNAseq = DNAseq_new2
                GC_CB[index].affinity = affinity(DNAseq_new2)
                GC_CB[index].cMyc -= 1
            
                    
            elif lethal2 and not lethal1: #one cell survive
                GC_DCB.append(GC_CB[index])
                GC_CB[index].DNAseq = DNAseq_new1
                GC_CB[index].affinity = affinity(DNAseq_new1)
                GC_CB[index].cMyc -= 1


            else: #both cell survive
                GC_CB.append(Cell(cMyc = GC_CB[index].cMyc-1, DNAseq = DNAseq_new1, ancestor = GC_CB[index].ancestor))
                GC_CB[index].DNAseq = DNAseq_new2
                GC_CB[index].affinity = affinity(DNAseq_new2)
                GC_CB[index].cMyc -= 1

                
                         
    elif mu == 1: #Centroblast migration:      CB -> CC
        
        index = int(len(GC_CB) * random.random())
        
        ndiv = GC_CB[index].cMyc
        rmigr_true = r_migration * (ndiv == 0)
            
        if rmigr_true >= r_migration*random.random():
            CellB = popcell(GC_CB,index)
            GC_CC.append(CellB)

            
    elif mu == 2: #Antigen uptake:     CC -> CC
        indexC = int(len(GC_CC) * random.random())
        GC_CC[indexC].pMHC = GC_CC[indexC].affinity
        
        
        
    elif mu == 3: #Centrocyte T-binding:   CC + TC = [CCTC]
        indexC = int(len(GC_CC) * random.random())
        if GC_CC[indexC].pMHC > 0:
            indexT = int(len(GC_TC) * random.random())
            CellB = popcell(GC_CC,indexC)
            GC_CCTC.append(CellB)
            popcell(GC_TC,indexT)
            GC_CCTC[-1].tstart = np.copy(t)
        
        
    elif mu == 4: #Tfh switch:             [CC1TC] + CC2 -> CC1 + [CC2TC]
        indexC1 = int(len(GC_CCTC) * random.random())
        indexC2 = int(len(GC_CC) * random.random())
        
        #Decide if the reaction happens depending on the affinities
        if GC_CC[indexC2].pMHC > GC_CCTC[indexC1].pMHC:
            CellB1 = GC_CCTC[indexC1]
            CellB2 = GC_CC[indexC2]
            GC_CCTC[indexC1] = CellB2
            GC_CC[indexC2] = CellB1
            
            GC_CCTC[indexC1].tstart = np.copy(t)
            GC_CC[indexC2].TChelp += (t - GC_CC[indexC2].tstart)*signal_strenght(GC_CC[indexC2].affinity,t)
            
            
    elif mu == 5: #Centrocyte unbinding:      #CCTC -> CCsel + TC
        index = int(len(GC_CCTC) * random.random())
        CellB = popcell(GC_CCTC,index)
        GC_CCsel.append(CellB) 
        GC_TC.append(Cell()) 
        
        GC_CCsel[-1].TChelp += (t - GC_CCsel[-1].tstart)*signal_strenght(GC_CCsel[-1].affinity,t)
        
        
    elif mu == 6: #Centrocyte apoptosis
        index = int(len(GC_CC) * random.random())
        CellB = popcell(GC_CC,index)
        GC_DCC.append(CellB)
        
        
    elif mu == 7: #Centrocyte recirculation
        index = int(len(GC_CCsel) * random.random())
        CellB = popcell(GC_CCsel,index)
        
        Thelp = CellB.TChelp #time in h
        aff = CellB.affinity
        n_Thelp.append([t,Thelp, aff, True])
        p_Tfh = np.argmin(np.abs(Tfh_percentil - Thelp))
        GC_CB.append(CellB)
          
        ndiv = min(int(6*p_Tfh/100) + 1,6) #we dont want 7 division in case we are on the 100th percentile
        GC_CB[-1].cMyc = ndiv
        GC_CB[-1].pMHC = 0
        GC_CB[-1].TChelp = 0
        
        
    elif mu == 8: #centrocyte exit
        
        index = int(len(GC_CCsel) * random.random())
        CellB = popcell(GC_CCsel,index)
        aff = CellB.affinity
    
        is_higher_treshold = (aff > antigenthreshold)
        if is_higher_treshold:  #plasma cell
            GC_PC.append(CellB)
        else:
            GC_MC.append(CellB)
                        
    else:
        print(" Warning, wrong reaction chosen for mu = %s" % mu)
        



def signal_strenght(affinity,t):
    #need to adjust the n
    n = 0
    signal = np.exp(np.power(affinity,n)) - 1
    return signal
    
    

def popcell(Cell_list, index):
    """
    exchange values of the element we want to delete with the last element, and then use pop()
    (popping the element i becomes O(1) rather than O(n))
    """
    
    Cell_last = Cell_list[-1]
    Cell_list[-1] = Cell_list[index]
    Cell_list[index] = Cell_last
    popped_cell = Cell_list.pop()
    
    return popped_cell


def SHM_mutation(DNAseq,pshm,delta,g,s):
    
    """
    Returns the sequences matrice after a mutation process (if mutated)
    Each mutation is lethal with probability delta
    
    """
    
    
    newDNA = np.copy(DNAseq)
    lethal = False
    
    if pshm <= 0:
        return newDNA, 0, lethal
    
    #Draw the number of mutation from binomial distribution
    NBCR = 660
    nmut = np.random.binomial(NBCR,pshm)
    
    if nmut == 0:
        return newDNA, nmut, lethal
    
    else:
        for n in range(nmut):
            
            #Check the fate of the mutation
            
            if random.random() < delta: #mutation is lethal
                lethal = True
                return newDNA, nmut, lethal
            
            elif random.random() < s/(1-delta): #mutation is silent
                pass
            
            else: #mutation changes the affinity
                index = np.random.randint(0,len(DNAseq))
                modif = np.random.randint(1,g)
                newDNA[index] = (DNAseq[index]+modif)%g
            
    return newDNA, nmut, lethal



def set_Tcell(GCdata, rhoTC,t):
    GC_CB,GC_CC,GC_CCsel,GC_CCTC,GC_TC,GC_MC,GC_PC,GC_DCC,GC_DCB,_ = GCdata
    
    NCC = len(GC_CC)
    NCT = int(NCC*rhoTC)
    NCT_current = len(GC_CCTC) + len(GC_TC)
    
    #print(NCT)
    #print(NCC,NCT_current)
    

    while NCT_current != NCT:
        if NCT_current < NCT:
            GC_TC.append(Cell())
            NCT_current += 1
        elif NCT_current > NCT:
            if len(GC_TC) > 0:
                GC_TC.pop()
            else:
                GC_CCTC.pop()
            NCT_current -= 1
            
            
            
            
def get_percentil(Tfh):
    
    y_tfh = np.array(Tfh[:,1])
    Tfh_percentil = np.zeros(100)
    for i in range(100):
        Tfh_percentil[i] = np.percentile(y_tfh, i)
    
    return Tfh_percentil
                