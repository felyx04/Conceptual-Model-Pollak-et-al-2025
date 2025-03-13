#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:18:52 2024

@author: pollakf

Note: - Same as ABR3_jax.py
      - CURRENT STANDARD ABR MODEL!
      
Note: Runs with float64 precision

"""


import numpy as np 
import matplotlib.pyplot as plt
from numba import njit
import scipy
import pickle
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import pandas as pd
from astropy.timeseries import LombScargle

# include package with color schemes
import MyModules.tol_colors as tc


###############################################################################
# BERENDS SEA LEVEL DATA
###############################################################################

###############################################################################
# Berends [-2 Myr - 0], no gap, no intersections
###############################################################################

# StartPosition = (-0.27792327678318857, -0.42242549904765475, 0.556488722819439, 0.9260783951764328, 0.2683280675474693, 5.657061668826499, 18.83018903627986, 0.001383447704904403, 10.823622546565844, 99.8543221447895, 5.859572628362376, 13.05501552548941, 1094.4619269266886, 47.15808573896594, 14.180557285806534)
# RMSE = 12.066875823051115

###############################################################################
# Berends [-2.6 Myr - 0], no gap, no intersections
###############################################################################

# Parameters from Pollak et al. (2025)
StartPosition = (0.05446396231274959, -0.23226119860247257, 0.805844411509879, 0.8640893077770145, -0.8056124360631497, 7.015599890396892, 18.894704532253378, 3.7642326067390965, 6.313335379159031, 96.77432851123312, -1.7410524406406258, 17.9953645441084, 1246.0466300974895, 10.598382705195123, -113.28842719672883)
# RMSE = 12.238498335305191

# Improved parameters
# StartPosition = (0.05462717275596418, -0.125699220612697, 0.6780006465590382, 0.8732096496353279, -0.5965564986673826, 6.433965901981196, 20.31087287061098, -2.70926489956787, 7.314722069040613, 97.80332128274688, -0.4858794951441894, 10.28418218763801, 1161.9451669041227, 10.757189669481988, 991.1311955153255)
# RMSE = 12.070902914390953

###############################################################################
# Berends [-3.6 Myr - 0], no gap, no intersections
###############################################################################

# StartPosition = (0.0064730553958807354, -0.21847312321001902, 0.300968750628199, 0.8573512055168512, -0.7505688789425305, 8.847477023046906, 26.588568484912088, 1.8616407879709673, 7.419679656471256, 100.20968147963852, -3.6724094949075563, -13.277541542159568, 1163.4910105243505, -2.221981262808366, -61.65504044348573)
# RMSE = 12.269776124153534


###############################################################################
# ROHLING SEA LEVEL DATA
###############################################################################

### StartParams for 1kyr resolution and interval [-2.6 Myr, 0 Myr] 
# StartPosition = (0.47427271380882335, 0.47742731331909727, 0.8807651465431242, 0.8768689118708445, -0.793522838090032, 9.836936134043295, 11.367141843008731, 14.853472963358072, 3.981223763794041, 95.0859752417596, -5.3440948545473885, 25.483730916936402, 1097.6985199220053, 28.61043357847331, -171.3688723420484)
# RMSE = 12.428879276939583


# Parameters = [aEsi, aEco, aO, ag, ad, taud, kEsi, kEco, kO, v0, v1, vi, ageMPT, v0MPT, taud_MPT]
#                0      1    2   3   4   5      6     7     8   9  10  11    12     13,     14
   
   
# Set resolution of model
resolution = 1_000

# Start year in kyr 
start_year = -int(2_600)

# Future simulation years in kyr (between 0 and 1_000)
future_time = int(250)

# Set title for Plot
title = 'ABR experiment'

# Set sea-level data: either Berends or Rohling (LR04 based + tuned age) 
sea_level_data = 'Rohling'

# save data in binary file with key:value pairs
# save_data = '../Data/ABR3_jax_2,6Myr.pkl'
save_data = None


time_steps = int((-start_year+future_time)*1e3/resolution)  # number of timesteps in model (with steps in future)

print(f"Model resolution: {resolution} years")
print(f"Number of timesteps: {time_steps}")   
print(f"Simulated time interval: {start_year} kyr BP - {future_time} kyr")   
print(f"Sea-level data from {sea_level_data} et al.\n")

#####################################################################
# Normalise a distribution    
@njit
def normalise(sample, length):
    # length is needed, to only include data from interval [start_year,0], but not futute data into mean
    m = np.mean(sample[:length])
    std = np.std(sample[:length])
    norm = (sample-m)/std
    return norm


#####################################################################
#Interpolate to artificially increase resolution (multiply by fac the number of points)
@njit
def interpol(sample, fac):
    new_sample = []
    tab = range(fac)
    for j in range(len(sample)):
        difference = sample[j+1]-sample[j]
        new_difference = difference/fac
        for x in tab:
            new_sample.append(sample[j]+x*new_difference)
        if j == len(sample)-2:
            break
    new_sample.append(sample[len(sample)-1]) #specific case for the last point
    return new_sample


#######################################################
#Compute the derivative for each time step

def Phi(i, v):
    
    # full simulation time
    sim_time = abs(start_year)+abs(future_time)
    
    # current time t
    t = -1 * start_year - ((i * sim_time / time_steps) / 2)
    
    # taud_t=taud after MPT and taud_t=taud_MPT before MPT + exclude division by 0
    taud_t = np.where(t >= ageMPT, taud_MPT, taud)
    
            
    if S[0] == "g" :
        dvdt = -aEsi*Esi[i]-aEco*Eco[i]-aO*EnO[i]+ag
   
    else :
        dvdt = -aEsi*Esi[i]-aEco*Eco[i]-aO*EnO[i]+ad-v/taud_t  
    
    return dvdt

#######################################################
 # calculate lower and upper bound for v, where deglaciation/glaciation starts
def calc_bounds(time_steps):
    v0s = np.zeros(2*time_steps+1)
    
    # complete simulation time
    sim_time = abs(start_year)+abs(future_time)
    
    for i in range(2*time_steps+1):
        # current time t
        t = -1 * start_year - ((i * sim_time / time_steps) / 2)
        
        # Before MPT
        if t >= ageMPT:
            v0_t = v0MPT
            
        # After MPT
        else:
            v0_t = v0
        
        v0s[i] = v0_t
   
    v0_bounds = v0s - kEsi*Esi - kEco*Eco - kO*EnO
    v1_bounds = kEsi*Esi + kEco*Eco + kO*EnO
    
    time_bounds = np.arange(-start_year*1e3, -future_time*1e3-1, -resolution*0.5)*1e-3
    return (time_bounds, v0s, v0_bounds, v1_bounds)

##########################################################
#Compute the modelled volume for the best parameters using the Runge–Kutta 4th order method

def modelledVolume(start_year, future_time, vi, n) :
    v = np.zeros(n+1)
    v[0] = vi
    state.append(S[1])
    
    # complete simulation time
    sim_time = abs(start_year)+abs(future_time)
    
    step = (future_time-start_year)/float(n)
    for i in range(n) :
        # current time t
        t = -1 * start_year - (i * sim_time / time_steps)
        
        # thresholds for state changes (use Esi, Eco, EnO at full time steps only)
        test_threshold_gd = kEsi*Esi[2*i]+kEco*Eco[2*i]+kO*EnO[2*i]+v[i]
        test_threshold_dg = kEsi*Esi[2*i]+kEco*Eco[2*i]+kO*EnO[2*i]
        
        # v0_t=v0 after MPT and v0_t=v0_MPT before MPT
        v0_t = np.where(t >= ageMPT, v0MPT, v0)
        
        # check for state changes
        if S[0] == "g":
            if test_threshold_gd > v0_t and test_threshold_dg > v1:
                S[0] = "d"
                S[1] = i
                
        else :
            if test_threshold_dg < v1 and test_threshold_gd < v0_t:
                S[0] = "g"
                Term_duration = (i-S[1])*sim_time/time_steps #Compute the duration of a termination
                Term_start = (abs(start_year)-S[1])*sim_time/time_steps #Compute the start of a termination
                ListDuration.append(Term_duration)
                ListStart.append(Term_start)
        
        if S[0]=="g":
            state.append(0)
        else:
            state.append(1)
            
        k1 = Phi(2*i,v[i])
        k2 = Phi(2*i+1,v[i]+k1*step/2.)
        k3 = Phi(2*i+1,v[i]+k2*step/2.)
        k4 = Phi(2*i+2,v[i]+step*k3)
        v[i+1] = v[i] + step/6.*(k1+2*k2+2*k3+k4)
    return v


#####################################################################
# Interpolates given data and time array linearly
# A new time array is created in interval (-start_year,-future_time) with given resolution (in years)
# e.g. [2000, ..., -1000] for interval 2Myr BP - 1Myr in future
def np_interpolation(array, name, resolution, time, start_year=-start_year, future_time=future_time, sea_data='Berends'): 
    if start_year>=3_600 or start_year<0:
        raise ValueError('start_year must be between 3_599 and 0!')
        
    # For interval [<3.6 Myr BP, <=2Myr future]
    # sea data
    if name=='sea':
        # default resolution for Brends: 100yr
        if sea_data=='Berends':
            if resolution==100:
                print('Berends sea-level data: Resolution set to default. Skipping interpolation step!')
                return (time, array)  
            else:
                # create new time array 
                print('Berends sea-level data: Resolution interpolated.')
                new_time = np.arange(0, start_year*1e3+1, resolution)*1e-3
                
                new_array = np.interp(new_time, np.flip(time), np.flip(array))
                
                return (np.flip(new_time), np.flip(new_array))
        
        # default resolution for Rohling: 1kyr
        elif sea_data=='Rohling':
            if resolution==1000:
                print('Rohling sea-level data: Resolution set to default. Skipping interpolation step!')
                return (time, array)  
            else:
                # create new time array 
                print('Rohling sea-level data: Resolution interpolated.')
                new_time = np.arange(0, start_year*1e3+1, resolution)*1e-3
                
                new_array = np.interp(new_time, np.flip(time), np.flip(array))
                
                return (np.flip(new_time), np.flip(new_array))
            
        else: 
            raise ValueError("sea_level_data must be either 'Berends' or 'Rohling'!")
    
    # orbital data
    else:
        # default resolution of loaded data is 1kyr -> skip this procedure
        if resolution==1000:
            print('Laska data: Resolution set to default. Skipping interpolation step!')
            return (time, array)  
        
        else:
            # create new time array 
            print('Laska data: Resolution interpolated')
            new_time = np.arange(-future_time*1e3, start_year*1e3+1, resolution)*1e-3
            new_array = np.interp(new_time, np.flip(time), np.flip(array))
            
            return (np.flip(new_time), np.flip(new_array))                                    

                
##########################################################
# Function for calculating the Bayesian Information Criterion (BIC)
# BIC = -2*LogLikelihood + N_Params*ln(N_DataPoints)                             
def calc_BIC(params, sea, sea_model):
    # use only reconstructed sea ice until present, not future predictions
    sea_model = sea_model[:len(sea)]
    
    # Number of data points
    N = len(sea)
    
    # Number of parameters
    n_params = len(params)
    
    # calculate log likelihood
    sea_std = np.std(sea)
    LogLikelihood = -0.5 * np.sum(np.square((sea-sea_model)/sea_std))
    
    # BIC
    BIC = -2*LogLikelihood + n_params*np.log(N)
    
    return BIC


##########################################################
# symmetric mean absolute percentage error (SMAPE)  [%]  
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))


########################################################

#Best parameters
aEsi = StartPosition[0]
aEco = StartPosition[1]
aO = StartPosition[2]
ag = StartPosition[3]
ad = StartPosition[4]
taud = StartPosition[5]
kEsi = StartPosition[6]
kEco = StartPosition[7]
kO = StartPosition[8]
v0 = StartPosition[9]
v1 = StartPosition[10]
vi = StartPosition[11]
ageMPT = StartPosition[12]
v0MPT = StartPosition[13]
taud_MPT = StartPosition[14]
    
S = ["g",0]


#Lists initialization
state=[]
v=[]
dvdt=[]
ListDuration=[]
ListStart=[]

####################################################################
# Load sea data from Berends et.al. 2020 [100yr resolution]
if sea_level_data=='Berends':
    data_sea = np.loadtxt('../Data/Berends_etal_2020_CP_supplement.dat', skiprows=10)
    time_sea = data_sea[:,0]
    mask_interval = np.where(np.logical_and(time_sea>=start_year, 
                                            time_sea<=0))
    time_sea = -1*time_sea[mask_interval]
    sea = -1*data_sea[:,1][mask_interval]

# Load sea data from Rohling et.al. 2022 (LR04 based + tuned age) [1kyr resolution]
elif sea_level_data=='Rohling':
    data_sea = pd.read_excel('../Data/Data summary sheet Rohling et al_Reviews of Geophysics 2022-v2.xlsx', skiprows=3)
    time_sea = data_sea.iloc[:,36]   # Tuned age for LR04 based solution of Rohling

    # Get only data for desired interval
    mask_interval = np.where(np.logical_and(time_sea>=start_year, 
                                            time_sea<=0))
    
    time_sea = -1*time_sea.iloc[mask_interval]
    sea = -1*data_sea.iloc[:,41].iloc[mask_interval]    # LR04 based solution
    time_sea = time_sea.to_numpy()
    sea = sea.to_numpy()
    
else:
    raise ValueError("sea_level_data must be either 'Berends' or 'Rohling'!")


# Load orbital data from Laska
data_orbital = np.loadtxt('../Data/Orbital_Params_-3,6MA-2MA_1kyr_steps.txt')
time = data_orbital[:,0]
mask_interval = np.where(np.logical_and(time>=start_year, 
                                        time<=future_time))
time = -1*time[mask_interval]
esinomega = -1*data_orbital[:,1][mask_interval]
ecosomega = -1*data_orbital[:,2][mask_interval]
O = data_orbital[:,3][mask_interval]
    

####################################################################
# Interpolate or choose data accordingly to set resolution (Default resolution = Resolution of loaded data = 100 years)
time_sea, sea = np_interpolation(sea, 'sea', resolution, time_sea, sea_data=sea_level_data)  
# sea_std = np_interpolation(sea_std, 'sea', resolution, time_sea, sea_data=sea_level_data)[1]  
# sea_std = np.where(sea_std<1, 1, sea_std)
esinomega = np_interpolation(esinomega, 'esinomega', resolution, time, sea_data=sea_level_data)[1]
ecosomega = np_interpolation(ecosomega, 'ecosomega', resolution, time, sea_data=sea_level_data)[1]
time, O = np_interpolation(O, 'O', resolution, time, sea_data=sea_level_data)

#####################################################################
#Normalization of parameters input
EnO = normalise(O, length=len(sea))
Esi = normalise(esinomega, length=len(sea))
Eco = normalise(ecosomega, length=len(sea))

#Interpolation to get data at the time step of 500 years (for half-step Runge-Kutta computation)
Esi = np.array(interpol(Esi,2))
Eco = np.array(interpol(Eco,2))
EnO = np.array(interpol(EnO,2))


##########################################################
#Modelling of the ice volume for the best parameters fit

icevolume = modelledVolume(start_year, future_time, vi, time_steps)

#calcul de l'écart modele donnees a chaque pas de temps
residuals = []
sum_residuals = 0
for i in range (len(sea)):
    sum_residuals = sum_residuals + (sea[i]-icevolume[i])**2
    residuals.append(sea[i]-icevolume[i])
    

# calculate bounds for deglaciation/glaciation
time_bounds, v0s, v0_bounds, v1_bounds = calc_bounds(time_steps)


#############################################################################################################################
##########################################################
# Spectral analysis: LombScargle Periodogram 
# Split into 2 diagrams: before and after 800 kyr
# step size in kyr
t_split = 800
step = (future_time-start_year)/time_steps

# index to the left of split start 
pre_split = int(np.floor(-start_year-t_split) / step)
print(f'Time of Split start: {time[pre_split]}')

time_sea_pre_split = time_sea[:pre_split+1]
ice_pre_split = icevolume[:pre_split+1]
sea_pre_split = sea[:pre_split+1]

time_sea_post_split = time_sea[pre_split+1:]
ice_post_split = icevolume[pre_split+1:len(sea)]
sea_post_split = sea[pre_split+1:len(sea)]

f_sea_LombScargle_pre_split, P_sea_LombScargle_pre_split = LombScargle(time_sea_pre_split, sea_pre_split, normalization='standard').autopower(minimum_frequency=1/150, maximum_frequency=1/10)
f_sea_LombScargle_post_split, P_sea_LombScargle_post_split = LombScargle(time_sea_post_split, sea_post_split, normalization='standard').autopower(minimum_frequency=1/150, maximum_frequency=1/10)
f_ice_LombScargle_pre_split, P_ice_LombScargle_pre_split = LombScargle(time_sea_pre_split, ice_pre_split, normalization='standard').autopower(minimum_frequency=1/150, maximum_frequency=1/10)
f_ice_LombScargle_post_split, P_ice_LombScargle_post_split = LombScargle(time_sea_post_split, ice_post_split, normalization='standard').autopower(minimum_frequency=1/150, maximum_frequency=1/10)


#########################################################
# calculate RMSE, MAE, R2, SMAPE and BIC 
# rmse = np.sqrt(np.sum(np.square(sea-icevolume[:len(sea)]))/len(sea))
rmse = root_mean_squared_error(y_true=sea, y_pred=icevolume[:len(sea)])
mae = mean_absolute_error(y_true=sea, y_pred=icevolume[:len(sea)])
R2 = r2_score(y_true=sea, y_pred=icevolume[:len(sea)])
SMAPE = smape(y_true=sea, y_pred=icevolume[:len(sea)])
BIC = calc_BIC(StartPosition, sea, icevolume)

# calculate perecntage of positive/negative residuls = over/underestimations of model
residuals = icevolume[:len(sea)]-sea
res = np.array(residuals)
res_pos = np.sum(res[res>=0])
res_neg = np.sum(res[res<0])
res_abs = np.sum(np.abs(res))



##########################################################
#outputs data    
    
print("      Minimum residuals = " + str((sum_residuals)))   
print("      Average of residuals = "+ str(((((sum_residuals)/len(sea)))**(1/2))))
print(f"      RMSE = {rmse}")
print(f"      MAE = {mae}")
print(f"      R² = {R2}")
print(f"      SMAPE = {SMAPE:.2f}%")
print(f"      BIC = {BIC}")
print(f'      Percentage of model overestimations: {res_pos/res_abs * 100:.2f}%')
print(f'      Percentage of model underestimations: {abs(res_neg)/res_abs * 100:.2f}%')
print("      Termination duration  : " + str(ListDuration))
print("      Start of termination : " + str(ListStart))


##########################################################
# FIGURE


# red-green safe color scheme
bright = tc.tol_cset('bright')

###############################################################################
# 1ST PLOT: Comparison model-data for the best fit StartPosition
fig, ax1 = plt.subplots(figsize=(20,5))
fig.tight_layout(pad=6)
ax1.plot(time_sea, sea, linestyle="--" , color=bright.blue, label="Berends data")
ax1.plot(time, icevolume, color=bright.purple, label="Model")
ax1.plot(time_bounds, v0s, linestyle="--" , color=bright.black, label=r"Deglaciation threshold: $v_0(t)$")

ax2 = ax1.twinx()
ax2.plot(time, state, linewidth = 0.8, color=bright.grey, label='Model state')

# plt.xticks([0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000])
# ax2.set_yticks([1,0], ['Interglacial', 'Glacial'], rotation=22.5)
ax2.set_yticks([1,0])

if start_year==-3_599:
    start_year=-3_600
ax1.set_xticks(np.arange(-start_year, -future_time-1, -200))


plt.xlim(-start_year,-future_time)
ax1.set_ylim(np.min([np.min(icevolume-10)]), 
             np.max([np.max(icevolume+10)]))
# plt.gca().invert_yaxis()
ax1.invert_yaxis()
ax1.set_xlabel("Age (ka)",weight='bold')
ax1.set_ylabel("Ice volume (m sl)",weight='bold')
ax2.set_ylabel('Model state')
fig.legend(ncol=3, loc=8)
if title!=None:
    plt.title(title+f'; RMSE={rmse:.2f} m', fontsize=18)
# plt.savefig('../Plots/ABR3.png', dpi=500, bbox_inches='tight')
plt.show()


###############################################################################
# 2ND PLOT: LombScargle Periodogram (Alternative with only 2 subplots)
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(12,6), dpi=300)
fig.tight_layout(pad=5.0)


ax[0].plot(1/f_sea_LombScargle_pre_split, P_sea_LombScargle_pre_split, linestyle='--', color=bright.blue, label="Berends")
ax[0].plot(1/f_ice_LombScargle_pre_split, P_ice_LombScargle_pre_split, color=bright.purple, label="Model")
# mark obliquity and precession cycles + 100 kyr
ax[0].vlines([100, 41, 23], 0, np.max([np.max(P_sea_LombScargle_pre_split), np.max(P_ice_LombScargle_pre_split)])+0.1, color='grey', linestyle='--')
# ax[0].set_ylim(0,np.max([np.max(P_sea_LombScargle_pre_split), np.max(P_ice_LombScargle_pre_split)])+0.1)
# ax[0].set_xlim([1/500,0.05])

ax[1].plot(1/f_sea_LombScargle_post_split, P_sea_LombScargle_post_split, linestyle='--', color=bright.blue, label="Berends")
ax[1].plot(1/f_ice_LombScargle_post_split, P_ice_LombScargle_post_split, color=bright.purple, label="Model")
# mark obliquity and precession cycles + 100 kyr
ax[1].vlines([100, 41, 23], 0, np.max([np.max(P_sea_LombScargle_post_split), np.max(P_ice_LombScargle_post_split)])+0.1, color='grey', linestyle='--')
# ax[1].set_ylim(0,np.max([np.max(P_sea_LombScargle_post_split), np.max(P_ice_LombScargle_post_split)])+0.1)
# ax[1].set_xlim([1/500,0.05])


ax[1].set_xlabel("Period [kyr]", weight='bold')
# Add a common y-axis label
fig.text(0.04, 0.5, r"Squared magnitude spectrum [$m^2$]", weight='bold', va='center', rotation='vertical')



for i in range(2):
    ax[i].text(0.485, 0.8, '100 kyr', color='grey', rotation=0, transform=ax[i].transAxes)
    ax[i].text(0.205, 0.8, '41 kyr', color='grey', rotation=0, transform=ax[i].transAxes)
    ax[i].text(0.115, 0.8, '23 kyr', color='grey', rotation=0, transform=ax[i].transAxes)

ax[0].legend()
ax[1].legend()
ax[0].set_title(f'{-start_year*1e-3:.1f}-{time[pre_split]*1e-3:.1f} Myr BP')
ax[1].set_title(f'{time[pre_split]*1e-3:.1f}-0 Myr BP')

# if title!=None:
#     plt.title(title, fontsize=25)
# plt.savefig('../Plots/ABR3_periodogram.png', dpi=300)
plt.show()



###############################################################################
# # 3RD PLOT: Comparison model-data for the best fit StartPosition
# fig, ax1 = plt.subplots(figsize=(20,4), dpi=300)
# ax1.plot(time, sea, linestyle="--" , color=bright.blue, label="Data")
# ax1.plot(time, icevolume, color=bright.purple, label="Model")
# ax1.plot(time_bounds, v0_bounds, color=bright.grey, label="Deglaciation threshold")
# ax1.plot(time_bounds, v0s, color=bright.grey, linestyle='--', label="v0")
# ax1.plot(time_bounds, v1_bounds, color=bright.black, alpha=.5, label="v1 bounds")
# ax1.axhline(v1, 0, 2000, linestyle='--', color=bright.black, alpha=.5, label="v1")
# ax1.invert_yaxis()

# ax2 = ax1.twinx()
# ax2.plot(time, state, linewidth = 0.8, color=bright.grey, label='Model state')

# plt.xticks([0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000])
# ax2.set_yticks([1,0])

# ax1.set_yticks([0,20,40,60,80,100,120])
# plt.xlim(2000,0)

# ax1.set_xlabel("Age (ka)",weight='bold')
# ax1.set_ylabel("Ice volume (m sl)",weight='bold')
# fig.legend()
# if title!=None:
#     plt.title(title, fontsize=25)
# plt.savefig(f'../Plots/ABR.png', dpi=300)
# plt.show()

###############################################################################

# #state of the model for the best fit StartPosition
# plt.figure(figsize=(20,4), dpi=300)
# plt.plot(time, state, "0.75", linewidth = 0.8, color=bright.black)
# plt.xticks([0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000])
# plt.xlim(0,2000)
# plt.yticks([0,1])
# plt.xlabel("Age (ka)",weight='bold')
# plt.ylabel("g or d (d=1, g=0)",weight='bold')
# plt.xlim(2000,0)
# plt.show()

# #residuals (model-data) for the best fit StartPosition
# plt.figure(figsize=(20,4), dpi=300)
# plt.plot(time, residuals, linestyle='--', color=bright.green, label="Model")
# plt.xticks([0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000])
# # plt.yticks([-40,-30,-20,-10,0,10,20,30,40])
# plt.xlim(2000,0)
# plt.xlabel("Age (ka)",weight='bold')
# plt.ylabel('Residuals (model-data)(msl)',weight='bold')
# plt.gca().invert_yaxis()
# plt.show()


###############################################################################
# Save data to binary file
if save_data!=None:
    # store all relevant data for later plotting
    data = {'rmse': rmse, 
            'sea': sea,
            'time_sea': time_sea,
            'time': time, 
            'start_year': start_year, 
            'future_time': future_time, 
            'time_steps': time_steps, 
            'resolution': resolution, 
            'sea': sea, 
            'icevolume': icevolume, 
            'v0s': v0s, 
            'v0_bounds': v0_bounds, 
            'v1_bounds': v1_bounds, 
            'time_bounds': time_bounds,
            'StartPosition': StartPosition, 
            'v0': v0, 
            'v0MPT': v0MPT, 
            'taud': taud, 
            'taud_MPT': taud_MPT, 
            'ageMPT': ageMPT, 
            'v1': v1, 
            'state': state, 
            'pre_split': pre_split, 
            'time_sea_pre_split': time_sea_pre_split, 
            'time_sea_post_split': time_sea_post_split, 
            'f_sea_LombScargle_pre_split': f_sea_LombScargle_pre_split,
            'P_sea_LombScargle_pre_split': P_sea_LombScargle_pre_split, 
            'f_sea_LombScargle_post_split': f_sea_LombScargle_post_split, 
            'P_sea_LombScargle_post_split': P_sea_LombScargle_post_split, 
            'f_ice_LombScargle_pre_split': f_ice_LombScargle_pre_split, 
            'P_ice_LombScargle_pre_split': P_ice_LombScargle_pre_split, 
            'f_ice_LombScargle_post_split': f_ice_LombScargle_post_split, 
            'P_ice_LombScargle_post_split': P_ice_LombScargle_post_split}
    
    # Write to binary file
    with open(save_data, 'wb') as f:
        pickle.dump(data, f)