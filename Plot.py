
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import random_sample as random
import scipy.interpolate as ip
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

from numpy import genfromtxt
from libAffinity import affinity



def plot_results(population, properties, Tend):
       
    #GC kinetics
    plot_GCpopulation(population,Tend)
    #clonal diversity
    n_clones = properties[:,:,1]
    NDS = properties[:,:,2]
    plot_clones(n_clones, NDS, Tend)
    
    #affinity
    affinity = properties[:,:,0]
    plot_affinity(affinity, Tend)
    

  
    
def plot_clones(n_clones, NDS, Tend):
    
    N_simulations = n_clones.shape[0]
    timepoints = n_clones.shape[1]
    time_points = np.linspace(0, Tend, timepoints)
    lwm = 3
    
    #compute mean clones and subclones removing zeros
    mean_n_clones = np.zeros(n_clones.shape[1])
    for i in range(mean_n_clones.shape[0]):
        if np.size(np.where(n_clones[:,i] != 0)) > 0:
            mean_n_clones[i] = n_clones[:,i].mean()/np.size(np.where(n_clones[:,i] != 0))*n_clones.shape[0]
    
    
    plt.figure(figsize=(11,5))
    for i in range(N_simulations): # Plot clones of CC
        plt.plot(time_points/24, n_clones[i,:]*1, 'k-', lw=0.3, alpha=0.2,color=sns.color_palette()[0])
    plt.plot(time_points/24, mean_n_clones, 'r-', lw=lwm, color='blue', label = "Model") # Plot affinity mean
    plt.ylim([0,np.max(mean_n_clones)*1.05])
    plt.xlabel('Time after immunization [day]')
    plt.ylabel('Number of clones in GC LZ')
    plt.xlim([0,42])
    plt.legend()
    plt.savefig("Results/ClonalDiversity.png")
    plt.show()
    
    
    
    
    
    #removing zero score for NDS average
    mean_NDS = np.zeros(NDS.shape[1])
    for i in range(mean_NDS.shape[0]):
        if np.size(np.where(NDS[:,i] != 0)) > 0:
            #mean_NDS[i] = NDS[:,i].mean()/np.size(np.where(NDS[:,i] != 0))*NDS.shape[0]
            mean_NDS[i] = np.median(NDS[:,i])/np.size(np.where(NDS[:,i] != 0))*NDS.shape[0]
            
            
            
    filename = 'Exp_data/GC_diversity.xlsx'
    df = pd.read_excel (filename, sheet_name=['AID-Confetti','AID-Confetti B1-8'], headers=None)
    dates = np.array([3,5,7,11,15,19,23])+5
    NDS_exp = [[] for i in range(len(dates))]
    
    days = np.array(df['AID-Confetti'].values[:,19][2:])+5
    NDS_ = np.array(df['AID-Confetti'].values[:,22][2:])
    DZdensity = np.array(df['AID-Confetti'].values[:,20][2:])
    nkeep = np.where(DZdensity >= 0.2)
    NDS_ = NDS_[nkeep]
    days = days[nkeep]
    
    for i in range(len(dates)):
        for j in np.where(days == dates[i]):
            NDS_exp[i].append(NDS_[j])
            
            
    d_start = 8
    istart = np.argmin(np.abs(time_points-d_start*24))
    plt.figure(figsize=(11,5))  
    for i in range(N_simulations): # Plot NDS of CC
        plt.plot(time_points[istart:]/24, NDS[i,istart:]*1, 'k-', lw=0.3, alpha=0.2,color=sns.color_palette()[0])
    plt.plot(time_points[istart:]/24, mean_NDS[istart:], 'r-', lw=lwm, color='blue', label = "Model")
    plt.xlabel('Time after immunization [day]')
    plt.ylabel('Normalized dominance score (NDS)')
    plt.xlim([0,42])
    
    av = []
    for i in range(len(NDS_exp)):
        for j in range(len(NDS_exp[i][0])):
            plt.scatter(dates[i],NDS_exp[i][0][j], color = "black",s=12)
        avi = np.median(NDS_exp[i])
        plt.plot([dates[i]-0.6,dates[i]+0.6],[avi,avi], color = 'black', lw = 4)
        av.append(avi)
    plt.plot([50,53],[avi,avi], color = 'black', lw = 5, label = "[Tas & al., 2016]") #just for the label
    plt.xlabel("Days after immunization")
    plt.legend()
    plt.ylim(0,1.1)
    plt.savefig("Results/DominanceScore.png")
    plt.show()      
    
    
    y_true = av
    y_pred0 = mean_NDS
    y_pred = np.zeros(len(y_true))
    for i in range(len(y_true)):
        ti = np.argmin(np.abs(time_points - days[i]*24))
        y_pred[i] = y_pred0[ti]
        
    try:
        rsquared = r2_score(y_true, y_pred)
        MSE = mean_squared_error(y_true, y_pred)

        NDS_std = [np.std(NDS[i]) for i in range(len(NDS))]
        NRMSE = np.sqrt(MSE/np.mean(NDS_std))
        print("  rsquared coef is %.2f" % rsquared)
        print("  Normalized RMSE is %.2f" % NRMSE)
    except:
        pass      




      
    
    
    
def plot_affinity(affinity, Tend):
    
    N_simulations = affinity.shape[0]
    timepoints = affinity.shape[1]
    time_points = np.linspace(0, Tend, timepoints)
    lwm = 3
    
    d_start = 5
    istart = np.argmin(np.abs(time_points-d_start*24))
    
    
    #Mean affinity (removing zeros)
    plt.figure(figsize=(11,5))
    mean_affinity = np.zeros(affinity.shape[1])
    for i in range(mean_affinity.shape[0]):
        if np.size(np.where(affinity[:,i] != 0)) > 0:
            mean_affinity[i] = affinity[:,i].mean()/np.size(np.where(affinity[:,i] != 0))*affinity.shape[0]
    
    for i in range(N_simulations): # Plot affinity of CC
        plt.plot(time_points[istart:]/24, affinity[i,istart:]*1, 'k-', lw=0.3, alpha=0.2,color=sns.color_palette()[0])
    plt.plot(time_points[istart:]/24, mean_affinity[istart:], 'r-', lw=lwm, color='purple') # Plot affinity mean
    plt.xlabel('Time after immunization [day]')
    plt.ylabel('Average affinity of B cell in GC')
    plt.ylim((0.35,0.73))
    plt.xlim([0,42])
    plt.savefig("Results/AffinityMaturation.png")
    plt.show()
    
    
    
    
def plot_GCpopulation(population, Tend):
        
    # ----------------- Plot the GC population --------------------------------------
    
    N_simulations = population.shape[0]
    timepoints = population.shape[1]
    
    time_points = np.linspace(0, Tend, timepoints)
    lwm = 3
    
    
    #GC knetics
    Data = np.load("Exp_data/GC_kinetics.npy")
    
    #GC volume from area
    days = Data[0,:]
    GC_volume = Data[1,:]
    GC_volume_min = Data[2,:]
    GC_volume_max = Data[3,:]

    plt.figure(figsize=(11,5))
    for i in range(N_simulations): # Plot CC+CB trajectories
        plt.plot(time_points/24, (population[i,:,0] + population[i,:,1] + population[i,:,2]), 'k-', lw=0.3, alpha=0.2,color=sns.color_palette()[0])
    plt.plot(time_points/24, population[:,:,0].mean(axis=0) + population[:,:,1].mean(axis=0) + population[:,:,2].mean(axis=0), 'r-', lw=lwm, color='red', label = "Model")
    plt.axvline(x=9,linestyle='--', linewidth=2, color='k')
    plt.axvline(x=4,linestyle='--', linewidth=2, color='k')
    plt.xlabel('Time after immunization [day]')
    plt.ylabel('Number of CB+CC')
    
    BC_volume = 1000
    plt.errorbar(days, GC_volume/BC_volume, yerr=[(GC_volume-GC_volume_min)/BC_volume , (GC_volume_max-GC_volume)/BC_volume], fmt='o', color = "black", ecolor='black',capsize=5, label = "[Wittenbrink & al., 2011]")
    plt.xlim([0,42])
    plt.legend()
    plt.savefig("Results/GCkin.png")
    plt.show()
    
    y_true = GC_volume/BC_volume
    y_pred0 = population[:,:,0].mean(axis=0) + population[:,:,1].mean(axis=0) + population[:,:,2].mean(axis=0)
    y_pred = np.zeros(len(y_true))
    for i in range(len(y_true)):
        ti = np.argmin(np.abs(time_points - days[i]*24))
        y_pred[i] = y_pred0[ti]
        
    try:
        rsquared = r2_score(y_true, y_pred)
        MSE = mean_squared_error(y_true, y_pred)
        y_std = (GC_volume_max-GC_volume_min)/BC_volume/2
        NRMSE = np.sqrt(MSE)/np.mean(y_std)
        print("  rsquared coef is %.2f" % rsquared)
        print("  Normalized RMSE is %.2f" % NRMSE)
    except:
        pass
    
    
    
    
    
 
    #DZ/LZ ratio
    r_av = Data[4,:]
    r_min = Data[5,:]
    r_max = Data[6,:]
    d_start = 9
    istart = np.argmin(np.abs(time_points-d_start*24))
          
    plt.figure(figsize=(11,5))
    for i in range(N_simulations): # Plot DZ/LZ ratio
        plt.plot(time_points[istart:]/24, population[i,istart:,0]/(population[i,istart:,1]+population[i,istart:,2]), 'k-', lw=0.3, alpha=0.2,color=sns.color_palette()[0])
    plt.plot(time_points[istart:]/24, population[:,istart:,0].mean(axis=0)/(population[:,istart:,1]+population[:,istart:,2]).mean(axis=0), 'r-', lw=lwm, color='red', label = "Model") # Plot CCap mean
    plt.axvline(x=9,linestyle='--', linewidth=2, color='k')
    plt.axvline(x=4,linestyle='--', linewidth=2, color='k')
    #plt.axhline(y=1,linestyle='--', linewidth=2, color='k')
    plt.xlim([0,42]) 
    
    plt.errorbar(days, r_av, yerr=[r_av-r_min, r_max-r_av], fmt='o', color = "black", ecolor='black',capsize=5, label = "[Wittenbrink & al., 2011]")
    plt.xlabel("Time after immunization [day]")
    plt.ylabel("DZ/LZ ratio")
    plt.legend()
    plt.yscale("log")
    plt.ylim([0.05,20]) 
    plt.savefig("Results/DZLZratio.png")
    plt.show()  
    
    
    y_true = r_av[3:]
    y_pred0 = population[:,:,0].mean(axis=0)/(population[:,:,1] + population[:,:,2]).mean(axis=0)-0.5
    y_pred = np.zeros(len(y_true))
    for i in range(len(y_true)):
        ti = np.argmin(np.abs(time_points - days[3:][i]*24))
        y_pred[i] = y_pred0[ti]
              
    try:
        
        rsquared = r2_score(y_true, y_pred)
        MSE = mean_squared_error(y_true, y_pred)
        y_std = (r_max-r_min)/2
        NRMSE = np.sqrt(MSE)/np.mean(y_std)
        print("  rsquared coef is %.2f" % rsquared)
        print("  Normalized RMSE is %.2f" % NRMSE)
    except:
        pass
    
    
    

    #MC production
    MC_IgM,days = construct_list_from_csv('Exp_data/Weisel_Immunity_2016_data_C.csv')
    MC_IgG,_ = construct_list_from_csv('Exp_data/Weisel_Immunity_2016_data_E.csv')
    
    MC = [(MC_IgM[i] + MC_IgG[i]) for i in range(min(len(MC_IgM),len(MC_IgG)))]
    
    
    MC_exp = [np.mean(MC[i]) for i in range(len(MC))]
    MC_sim = (np.gradient(population[:,:,5], axis = 1)/(population[:,:,1] + population[:,:,2])).mean(axis=0)
    nwhere = np.where(np.isfinite(MC_sim))
    coef = np.max(MC_exp)/np.max(MC_sim[nwhere])
    

    plt.figure(figsize=(11,5))
    for i in range(N_simulations): # Plot MC trajectories gradient
        plt.plot(time_points/24, np.gradient(population[i,:,5])/(population[i,:,1]+population[i,:,2])*coef, 'k-', lw=0.3, alpha=0.2,color=sns.color_palette()[0])
    plt.plot(time_points/24, (np.gradient(population[:,:,5], axis = 1)/(population[:,:,1] + population[:,:,2])).mean(axis=0)*coef, 'r-', lw=lwm, color='red', label = "Model")
    plt.xlabel('Time after immunization [day]')
    plt.ylabel('MC production per CC /h [a.u.]')
    plt.xlim([0,42])
    
    av = []
    for i in range(len(MC)):
        for j in range(len(MC[i])):
            plt.scatter(days[i],MC[i][j], color = "black",s=12)
        avi = np.median(MC[i])
        plt.plot([days[i]-0.6,days[i]+0.6],[avi,avi], color = 'black', lw = 4)
        av.append(avi)
    plt.plot([50,53],[avi,avi], color = 'black', lw = 5, label = "[Weisel & al., 2016]") #just for the label
    plt.xlabel("Days after immunization")
    plt.legend()
    plt.ylim(ymax=40)
    plt.yticks([])
    plt.savefig("Results/MC.png")
    plt.show()      
    
    
    y_true = av
    y_pred0 = (np.gradient(population[:,:,5], axis = 1)/(population[:,:,1]+population[:,:,2])).mean(axis=0)*coef
    y_pred = np.zeros(len(y_true))
    for i in range(len(y_true)):
        ti = np.argmin(np.abs(time_points - days[i]*24))
        y_pred[i] = y_pred0[ti]
        
    try:
        rsquared = r2_score(y_true[:-5], y_pred[:-5])
        MSE = mean_squared_error(y_true[:-5], y_pred[:-5])

        MC_std = [np.std(MC[i]) for i in range(len(MC))]
        NRMSE = np.sqrt(MSE)/np.mean(MC_std[:-5])
        print("  rsquared coef is %.2f" % rsquared)
        print("  Normalized RMSE is %.2f" % NRMSE)
    except:
        pass


    
    
    
    
    
    #PC production
    PC,days = construct_list_from_csv('Exp_data/Weisel_Immunity_2016_data_G.csv')
    
    PC_exp = np.array([np.mean(PC[i]) for i in range(len(PC))])
    PC_sim = (np.gradient(population[:,:,6], axis = 1)/(population[:,:,1]+population[:,:,2])).mean(axis=0)
    nwhere = np.where(np.isfinite(PC_sim))
    coef = np.max(PC_exp)/np.max(PC_sim[nwhere])
    
    
    #results from simulation
    plt.figure(figsize=(11,5))
    for i in range(N_simulations): # Plot PC trajectories gradient
        plt.plot(time_points/24, np.gradient(population[i,:,6])/(population[i,:,1]+population[i,:,2])*coef, 'k-', lw=0.3, alpha=0.2,color=sns.color_palette()[0])
    plt.plot(time_points/24, (np.gradient(population[:,:,6], axis = 1)/(population[:,:,1]+population[:,:,2])).mean(axis=0)*coef, 'r-', lw=lwm, color='red', label = "Model")
    plt.xlabel('Time after immunization [day]')
    plt.ylabel('PC production per CC /h [a.u.]')
    plt.xlim([0,42])
    
    #results from experiments
    av = []
    for i in range(len(PC)):
        for j in range(len(PC[i])):
            plt.scatter(days[i],PC[i][j], color = "black", s=12)
        avi = np.median(PC[i])
        plt.plot([days[i]-0.6,days[i]+0.6],[avi,avi], color = 'black', lw = 4)
        av.append(avi)
    plt.plot([50,53],[avi,avi], color = 'black', lw = 5, label = "[Weisel & al., 2016]")
    #plt.ylim([-1,40])
    #plt.ylabel("Cell count [%]")
    #plt.xlabel("Days after immunization")
    plt.ylim(ymax=20)
    plt.yticks([])
    plt.legend()
    plt.savefig("Results/PC.png")
    plt.show()   
    
    
    y_true = av
    y_pred0 = (np.gradient(population[:,:,6], axis = 1)/(population[:,:,1]+population[:,:,2])).mean(axis=0)*coef
    y_pred = np.zeros(len(y_true))
    for i in range(len(y_true)):
        ti = np.argmin(np.abs(time_points - days[i]*24))
        y_pred[i] = y_pred0[ti]

    try:
        rsquared = r2_score(y_true[:-5], y_pred[:-5])
        MSE = mean_squared_error(y_true[:-5], y_pred[:-5])
        PC_std = [np.std(PC[i]) for i in range(len(PC))]
        NRMSE = np.sqrt(MSE)/np.mean(PC_std[:-5])
        print("  rsquared coef is %.2f" % rsquared)
        print("  Normalized RMSE is %.2f" % NRMSE)
    except:
        pass
    
    

    
    
    #DC trajectories gradient
    d_start = 8
    istart = np.argmin(np.abs(time_points-d_start*24))
    
    plt.figure(figsize=(11,5))
    for i in range(N_simulations): 
        plt.plot(time_points[istart:]/24, np.gradient(population[i,istart:,7])/(population[i,istart:,1]+population[i,istart:,2])*100/6, 'k-', lw=0.3, alpha=0.2,color=sns.color_palette()[0])
    for i in range(N_simulations):
        plt.plot(time_points[istart:]/24, np.gradient(population[i,istart:,8])/population[i,istart:,0]*100/6, 'k-', lw=0.3, alpha=0.2,color=sns.color_palette()[0])    
    plt.plot(time_points[istart:]/24, (np.gradient(population[:,istart:,7], axis = 1)/(population[:,istart:,1]+population[:,istart:,2])).mean(axis=0)*100/6, 'r-', lw=lwm, color='gray', label = "Model LZ")
    plt.plot(time_points[istart:]/24, (np.gradient(population[:,istart:,8], axis = 1)/(population[:,istart:,0])).mean(axis=0)*100/6, 'r-', lw=lwm, color='black', label = "Model DZ")
    
    plt.hlines(8, d_start, 42, linestyle='--',color = "black", lw = 3, label = "[Mayer & al., 2017]")
    
    plt.legend()
    plt.xlabel('Time after immunization [day]')
    plt.ylabel('Cell Death / h [%]')
    plt.xlim([0,42])
    plt.ylim([0,20])
    plt.legend()
    plt.savefig("Results/CellDeath.png")
    plt.show()
    
    
    
    
def plot_clonal_competition(BCR, n_seeder, Tend):
    
    """
    Population:
        Array[t][cell_counts] containing the population of 6 GC cells
        N_CB, N_CC, N_CCTC, N_TC, N_MC, N_PC, N_DC, N_DB
      
    BCR:
        list[t][a][cell][seq] containing the BCRseq of each cell of each ancestors at each time step
    """
    
    max_clones = n_seeder
    nt = len(BCR)
    dom_score = np.empty((nt,n_seeder))
    dom_score[:] = np.NaN
    aff_clone = np.empty((nt,n_seeder))
    aff_clone[:] = np.NaN
    N_GC = np.zeros(nt)
    
    for t in range(nt):
        
        anc_count = np.array([len(BCR[t][a]) for a in range(n_seeder)])
        if np.sum(anc_count) >0:
            for a in range(n_seeder):
                if len(BCR[t][a]) > 0:
                    affinity_a = np.zeros(len(BCR[t][a]))
                    for c in range(len(BCR[t][a])):
                        affinity_a[c] = affinity(BCR[t][a][c])
                    aff_clone[t,a] = np.mean(affinity_a)
                    dom_score[t,a] = anc_count[a]/np.sum(anc_count)
                    N_GC[t] = np.sum(anc_count)
                    
    clone_score = np.empty((nt,n_seeder))
    clone_score[:] = np.NaN
    for t in range(nt):
        for a in range(n_seeder):
            if not np.isnan(dom_score[t,a]):
                clone_score[t,a] = dom_score[t,a]*aff_clone[t,a]
                

    
    time_points = np.linspace(0, Tend, nt)
    plot_clone_thresh = 0.05
    nplot = 7
    
    
    #clonal abundancy
    dom = np.zeros(n_seeder)
    for a in range(n_seeder):
        nwnan = np.where(np.isnan(dom_score[:,a]) == False)
        dom[a] = np.mean(dom_score[:,a][nwnan])
    
    array = np.array(-dom)
    order = array.argsort()
    rank = order.argsort()
    
    
    #clonal dominance
    plt.figure(figsize=(11,5))
    for a in range(n_seeder):
        nwnan = np.where(np.isnan(dom_score[:,a]) == False)
        #if np.mean(dom_score[:,a][nwnan]) > plot_clone_thresh:
        if rank[a] < nplot:
            plt.plot(time_points/24,dom_score[:,a])
    plt.ylabel("Clone dominance")
    plt.xlabel("Time after immunization [day]")
    plt.yscale("log")
    plt.xlim([0,40])
    plt.show()
    
    #affinity
    plt.figure(figsize=(11,5))
    for a in range(n_seeder):
        nwnan = np.where(np.isnan(dom_score[:,a]) == False)
        #if np.mean(dom_score[:,a][nwnan]) > plot_clone_thresh:
        if rank[a] < nplot:
            plt.plot(time_points/24,aff_clone[:,a])
    plt.ylabel("Clone average affinity")
    plt.xlabel("Time after immunization [day]")
    plt.xlim([0,40])
    plt.show()
    
    
    
    
    
    
    
def construct_list_from_csv(filename):
    
    cell_counts_ = genfromtxt(filename, delimiter=',')
    cell_counts = [[] for i in range(cell_counts_.shape[1])]
    for i in range(cell_counts_.shape[1]):
        for j in range(cell_counts_.shape[0]):
            if not np.isnan(cell_counts_[j,i]):
                cell_counts[i].append(cell_counts_[j,i])
                
    days = [7,10,13,16,19,22,25,28,37,31,34,40,43,46,49]
    
    return cell_counts,days