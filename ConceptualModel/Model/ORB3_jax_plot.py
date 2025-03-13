#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:31:30 2024

@author: pollakf

Note: - Same as ORB2_jax_plot.py
      - CURRENT STANDARD GRAD MODEL!
      
Note: Runs with float64 precision
       
"""


import numpy as np 
import matplotlib.pyplot as plt
from numba import njit
import scipy
import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from astropy.timeseries import LombScargle

# include package with color schemes
import MyModules.tol_colors as tc


###############################################################################
# BERENDS SEA LEVEL DATA
###############################################################################

###############################################################################
# Berends [-2 Myr - 0], no gap, no intersections
###############################################################################

# StartPosition = (-0.22333203005236868, -0.5275858239733733, 0.7847353179270993, 0.7156385586711167, -0.13629457635056497, 8.762566949325446, 22.532344499226724, 12.727544393982043, 9.86837866080998, 91.34280985107875, -0.9131655952168387, 8.426294445945047)
# RMSE = 16.017935878864165

###############################################################################
# Berends [-2.6 Myr - 0], no gap, no intersections
###############################################################################

# Pollak et al. 2025 parameters
StartPosition = (0.6079359447956998, -0.5483837180939726, 1.280062563717673, 0.3236585862102289, -0.2704773124262032, 1954.8437561723867, 583.8280419378767, 9198.791043285975, 567.0562060855045, -4598.836228871277, 4919.806354271822, 19.840953667695814)
# RMSE = 17.498554732440173

# improved solution
# StartPosition = (0.8612982740956081, -0.06731458233924315, 1.5734934833725323, -0.02863184302100308, 9.458985585523124, 5.761313044879671, -9.639368180195447, -6.020258904947923, -6.042187987113039, 57.187447925889614, -12.3365496331146, 16.220811250067555)
# RMSE = 17.024969297013794

###############################################################################
# Berends [-3.6 Myr - 0], no gap, no intersections
###############################################################################

# StartPosition = (0.4491084231508585, -0.20534646410571025, 1.0716619770816378, -0.10521793243655433, 0.4530369541264896, 130.5988771841212, 0.6620487533259658, 8.522534753921947, -5.419707716511914, 23.203071446105163, 5.441902865361001, -22.909144406409318)
# RMSE = 15.686186310643585


###############################################################################
# ROHLING SEA LEVEL DATA
###############################################################################

###############################################################################
# Rohling [-2.6 Myr - 0], no gap, no intersections
###############################################################################

# StartPosition = (0.45312169405625286, 0.9872605562181783, 0.2826387873084808, 0.6544828946867912, -0.5729356527770537, 7.924652361813096, 19.280608768896464, 2.8494754995527636, 8.601929812604856, 74.44865618081822, 1.971233267919818, -2.996671110601905)
# RMSE = 17.76374584953754



# Parameters = [aEsi, aEco, aO, ag, ad, tau_d, kEsi, kEco, kO, v0, v1, vi]
#                0      1    2   3   4     5     6     7    8   9  10  11

   
# Set resolution of model
resolution = 1_000

# Start year in kyr 
start_year = -int(2_600)

# Future simulation years in kyr (between 0 and 1_000)
future_time = int(250)

# Set title for Plot
title = 'ORB experiment'

# Set sea-level data: either Berends or Rohling (LR04 based + tuned age) 
sea_level_data = 'Berends'

# save data in binary file with key:value pairs
# save_data = '../Data/ORB2_jax_2,6Myr.pkl'
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
    if S[0] == "g" :
        dvdt = -aEsi*Esi[i]-aEco*Eco[i]-aO*EnO[i]+ag
   
    else :
        dvdt = -aEsi*Esi[i]-aEco*Eco[i]-aO*EnO[i]+ad-v/tau_d  
    
    return dvdt

#######################################################
 # calculate lower and upper bound for v, where deglaciation/glaciation starts
def calc_bounds(time_steps):
    v0s = v0*np.ones(2*time_steps+1)
    
    # complete simulation time
    sim_time = abs(start_year)+abs(future_time)
    
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
    for i in range(n):
        # thresholds for state changes (use Esi, Eco, EnO at full time steps only)
        test_threshold_gd = kEsi*Esi[2*i]+kEco*Eco[2*i]+kO*EnO[2*i]+v[i]
        test_threshold_dg = kEsi*Esi[2*i]+kEco*Eco[2*i]+kO*EnO[2*i]
        
        if S[0] == "g":
            if test_threshold_gd > v0 and test_threshold_dg > v1:
                S[0] = "d"
                S[1] = i
                
        else :
            if test_threshold_dg < v1 and test_threshold_gd < v0:
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
tau_d = StartPosition[5]
kEsi = StartPosition[6]
kEco = StartPosition[7]
kO = StartPosition[8]
v0 = StartPosition[9]
v1 = StartPosition[10]
vi = StartPosition[11]
    
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
# 1ST PLOT: Comparison model-data for the best fit BestParamPlot
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
ax1.set_xticks(np.arange(-start_year, -future_time-1, -200))

plt.xlim(-start_year,-future_time)
ax1.set_ylim(np.min([np.min(icevolume-10), np.min(sea-10)]), 
             np.max([np.max(icevolume+10), np.max(sea+10)]))
# plt.gca().invert_yaxis()
ax1.invert_yaxis()
ax1.set_xlabel("Age (ka)",weight='bold')
ax1.set_ylabel("Ice volume (m sl)",weight='bold')
ax2.set_ylabel('Model state')
fig.legend(ncol=3, loc=8)
if title!=None:
    plt.title(title+f'; RMSE={rmse:.2f} m', fontsize=18)
# plt.savefig('../Plots/ORB2.png', dpi=500, bbox_inches='tight')
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
# plt.savefig('../Plots/ORB2_periodogram.png', dpi=300)
plt.show()

###############################################################################
# # 3RD PLOT: Comparison model-data for the best fit BestParamPlot
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
# plt.savefig(f'../Plots/test2.png', dpi=300)
# plt.show()

###############################################################################

# #state of the model for the best fit BestParamPlot
# plt.figure(figsize=(20,4), dpi=300)
# plt.plot(time, state, "0.75", linewidth = 0.8, color=bright.black)
# plt.xticks([0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000])
# plt.xlim(0,2000)
# plt.yticks([0,1])
# plt.xlabel("Age (ka)",weight='bold')
# plt.ylabel("g or d (d=1, g=0)",weight='bold')
# plt.xlim(2000,0)
# plt.show()

# #residuals (model-data) for the best fit BestParamPlot
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