
import numpy as np
import random as rd
import matplotlib.pyplot as plt

from libGillespie import launch_Gillespie, initialize_GC, get_percentil
from libAffinity import get_affinity
from Plot import plot_results, plot_clonal_competition

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 


""" --------------------------------------------------------------------------------------------
 Implementation of the Germinal center simulation with a modified Gillepsie algorithm. 
 Adapted to account for individual properties of each reactants
 The simulation account for affinity dependant reaction rates
 
 [ref] ..........................................................
       

  In the simulation, 4 types of reactants are considered:
      - Centroblast (Dark Zone) = CB
      - Centrocytes (Light Zone) = CC
      - Binnded Centrocytes (Light Zone) = [CCTC]
      - Free T follicular helper (Light Zone) = Tfh
      
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
      
   The affinity between BCR and antigens is related to the similarities
   between their respective DNA sequences. More precisely, the affinity is
   computed as the Hamming distance between the antigen and BCR DNA seq.
   A higher affinity increases the likelihood of a T cell binding.
    
 --------- Created 14/08/2019 ------------------------------------------------------------------
"""



def main_simulation(N_simulations = 15):
    
    
    print("\n=================================================================================") 
    print("=================== GC Simulation in 3D - Gillepsie algorithm ===================") 
    print("=================================================================================\n\n")                 
    
    rd.seed(100)
    
    #main parameters
    precompute_percentile = False   #If you change the distribution degree, please set True to recompute the appropriate Tfh percentil
    ndays = 37
    Tend = 24*ndays                  #length of the simulation, in hours
    timepoints = int(ndays*4)
    
    

    # ------ intercellular GC dynamic related parameters ------------------------    
    r_activation        = 4.1         #0 -> CB GC seeding rate
    r_division          = 0.134       #CB -> 2CB Division rate
    r_migration         = 0.4         #CB -> CC migration rate
    r_apoptosis         = 0.084       #CC -> 0 apoptosis rate
    r_recirculate       = 3.75        #CCsel -> CB recirculation rate
    r_unbinding         = 2.000       #CCTC -> CC + TC       
    r_FDCencounter      = 1           #CC antigen uptake (should be 10)
    r_TCencounter       = 10          #CC + TC = [CCTC] Tcell encounter rate
    rhoTC               = 1/42        #TC:CC ratio
    p                   = 0.7         #CCsel recirculation probability
    r_exit = r_recirculate/p - r_recirculate
    
    rates = np.array([r_activation, r_division, r_migration, r_FDCencounter, r_TCencounter, r_unbinding, r_apoptosis, r_recirculate, r_exit, rhoTC])
    
    
    
    # ------ intracellular affinity related parameters ------------------------       
    Lseq               = 30       #IgV sequence length (number of sites)
    pshm               = 1e-3     #SHM mutation rate per site for max cMyc
    g_site             = 4        #Number of possibilities for each site (ACTG)
    antigenthreshold   = 0.45     #Beta threshold to decide if cell becomes MC or PC, arbitrary at the moment
    delta = 0.52                      #Probability of mutation being lethal
    s                  = 0.25         #Probability of mutation being silent (set to 0.25 arbitrarily)
    
    param_intra = [Lseq,g_site,pshm,antigenthreshold,s,delta]
    
    
    
    
    # ------ Parameters to initialize the GC ----------------------------------
    N_CB   = 0    #initial number of CB
    N_CC   = 0    #initial number of CCap
    N_CCsel= 0    #initial number of CCsel
    N_CCTC = 0    #initial number of CCTC
    N_TC   = 50   #initial number of TC
    N_MC   = 0    #initial number of MC
    N_PC   = 0    #initial number of PC
    N_DCC  = 0    #initial number of dead centrocyte
    N_DCB  = 0    #initial number of dead centroblast
    N_seeder = 1000   #number of naive seeder cells
    N_GC_init = [N_CB, N_CC, N_CCsel, N_CCTC, N_TC, N_MC, N_PC, N_DCC, N_DCB, N_seeder]
    
    initial_parameters = [N_GC_init, N_seeder]
    
    
    
    # ---- Run the Gillespie simulation several times and store the results ---
    population = np.empty((N_simulations, timepoints+1, len(N_GC_init)-1))
    properties =  np.empty((N_simulations, timepoints+1, 3))


    #Initiallize a list of reactants:
    print("   Initializing GC ...")
    GCdata_init, seederBCR = initialize_GC(initial_parameters,param_intra)
    
    #precompute Tfh percentil
    if precompute_percentile:
        print("  Precomputing Tfh distribution ...")
        print("    1/2...")
        Tfh_percentil = np.linspace(0,100,100)
        _,_,_,n_Thelp = launch_Gillespie(GCdata_init,Tend, rates, param_intra, timepoints, Tfh_percentil)
        Tfh_percentil = get_percentil(n_Thelp)
        print("    2/2...")
        _,_,_,n_Thelp = launch_Gillespie(GCdata_init,Tend, rates, param_intra, timepoints, Tfh_percentil)
        Tfh_percentil = get_percentil(n_Thelp)
        np.save("Tfh_percentil.npy",Tfh_percentil) 
        
        y_tfh = np.array(n_Thelp[:,1])
        plt.hist(y_tfh, bins = 50, color = "green", density=True)
        #plt.axvline(Thelpmin, lw = 2, linestyle= '--')
        plt.yscale('log')
        plt.xlabel("Cumlulated Tcell signal")
        plt.ylabel("Occurence")
        plt.show()
        
    else:
        Tfh_percentil = np.load("Tfh_percentil.npy")
        

        
    for i in range(N_simulations):
        print("   Simulation",i+1,"/",N_simulations,"...")
        population[i,:,:], properties[i,:,:], BCRs, _ = launch_Gillespie(GCdata_init,Tend, rates, param_intra, timepoints, Tfh_percentil)
        print("      Saving results ...")
        plot_clonal_competition(BCRs, N_seeder, Tend)
        
    final_results = {"population": population, "properties": properties}
    np.savez('Results/Simulation_results', **final_results)
    
    plot_results(population, properties, Tend)

    


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 18})
    main_simulation()