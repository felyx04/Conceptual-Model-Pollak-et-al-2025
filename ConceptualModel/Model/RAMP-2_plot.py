#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 10:33:51 2025

@author: pollakf

Note: - Same as RAMP97_jax_plot.py, but changed time according to Ganopolski
      - time runs in phys. units: -2600 -> 0  ; t1 and t2 also negative   ; changed times in ramp
      - I0(t), v0(t), v1(t) ramp-like (same ramp)
      - added GMSL (Clark, 2025) and removed ben d18O as targets
      - dropped -1* factor when loading esinw data
      -> this was used in the Legrain et al. model, but it changes sign of the actual esinw data
      -> new threshold: g->d): v*I +v >v0(t) 
                        d->g): I<I0(t) & v<v1(t)
      - No longer Ik and Ialpha, just one forcing I(t)
      - I(t) = aEsi*Esi + aO*O
      - no alpha_d
      - taud is constant in time
      - co-precession dropped
      - Bounds: params: +/- 10_000
      - 13 params
      
Note2: Runs with float64 precision
       
"""


import numpy as np 
import matplotlib.pyplot as plt
from numba import njit
import scipy
import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from scipy.stats import pearsonr
from astropy.timeseries import LombScargle
import pyleoclim as pyleo

# include package with color schemes
import MyModules.tol_colors as tc


###############################################################################
# BERENDS SEA LEVEL DATA
###############################################################################

###############################################################################
# Berends [-2 Myr - 0]
###############################################################################

# emcee + dynesty
# StartPosition = (-1, 1, 1, 1, 1, 1, 1, -1200, -800, 1, 1, 1, 1)
# StartPosition = (-0.43716450887423663, 0.3307099815379785, 0.9771809999630383, 9.515671961740894, 27.187633584065082, 48.67256682719099, 121.32230737342516, -1830.1128391332975, -562.9778341466616, 7426.47937177786, 3253.6001485350744, 10.181307134435361, 14.774082491567363)
# RMSE = 12.19529585391784

###############################################################################
# Berends [-2.6 Myr - 0]
###############################################################################

# emcee + dynesty
# StartPosition = (-1, 1, 1, 1, 1, 1, 1, -1200, -800, 1, 1, 1, 1)
# StartPosition = (-0.5090203501048705, 0.45348568853914983, 0.9815494358729182, 10.70514768679791, 27.36049061039328, 37.35138684387872, 131.9837204398457, -2178.920495835071, -463.92646192578957, -0.8342438977028693, 200, 7.9254734180704105, 16.349081614870425)
# RMSE = 11.956623711758223

# Clark solution
# StartPosition = (-1.136658501233768, 1.258269137570222, 1.7580058627236668, 6.538920642650687, 20.11557982395534, 188.76194902434904, 230.10607314461774, -2134.2316099904683, -745.9521175121465, 22.182319313783374, -0.44150013722874537, 50.5212340318005, 102.36954660799226)
# StartPosition = (-0.6472767202921449, 0.5061502177013214, 0.7384556821272855, 8.005889496065906, 43.64922399501382, 42.73763448262623, 126.95981458116827, -2050.5488235286552, -818.4711237796465, 8.023351514777714, 0.20851978749302563, 1.9365473822011272, 36.12066141423095)
# RMSE = 12.504973656349515


# StartPosition = (-0.5884169134374966, 0.5318247545292409, 0.8217986175738236, 7.194732231231515, 38.504040523227296, 42.107975558236696, 142.3857878945521, -2272.7141119990124, -31.239793423460917, 1.613503486041131, 0.11592898666437554, 0.584480971205182, 47.9581056570014)
# RMSE = 12.005262332490934

###############################################################################
# Berends [-3 Myr - 0]
###############################################################################

# Clark solution
# StartPosition = (-0.6472767202921449, 0.5061502177013214, 0.7384556821272855, 8.005889496065906, 43.64922399501382, 42.73763448262623, 126.95981458116827, -2050.5488235286552, -818.4711237796465, 8.023351514777714, 0.20851978749302563, 1.9365473822011272, 36.12066141423095)
# StartPosition = (-0.5957880385234549, 0.5056438714626447, 0.8078423953622473, 8.04230463638938, 7.081004253905572, 16.937266761702492, 140.028292263668, -2969.6973028512934, -25.120338704462668, 2.181900903319734, -0.017490984135115968, -9.753643203344433, 44.85068702753064)
# RMSE = 11.659358787225045

# 2.6 Myr solution
# StartPosition = (-0.5884169134374966, 0.5318247545292409, 0.8217986175738236, 7.194732231231515, 38.504040523227296, 42.107975558236696, 142.3857878945521, -2272.7141119990124, -31.239793423460917, 1.613503486041131, 0.11592898666437554, 0.584480971205182, 47.9581056570014)
# StartPosition = (-0.5906552776301797, 0.5024707574927784, 0.7961382269542748, 8.02793599456681, 9.839366389736236, 15.757935850100315, 135.7744316147947, -2992.2050161664556, -62.71772798023692, 2.239371275475545, 0.003837812858366485, -10.360393440389766, 43.87382534302778)
# RMSE = 11.656751863427248

###############################################################################
# Clark GMSL [-2 Myr - 0]
###############################################################################

# emcee + dynesty
# StartPosition = (-1, 1, 1, 1, 1, 1, 1, -1200, -800, 1, 1, 1, 1)
# StartPosition = (-0.8846463036594662, 0.8099000472630221, 1.7439812872930212, 2.465169838834367, 13.435869534158073, 239.8190978184128, 156.4314606635155, -1017.8989480394566, -920.8289821011772, 6187.411766201387, 1807.2601760067496, 58.20322149912396, 63.946612082213896)
# RMSE = 28.450392398018348

 
# 2.6 Myr solution
# StartPosition = (-1.1343938917360674, 1.257769829808808, 1.7459735984473355, 6.583527120891631, 19.739059337807475, 187.2822360887286, 228.88535071966464, -2000, -762.0781747874983, 25.98771908505603, -0.5157573150308963, 50.907066558484985, 102.93120460213129)
# StartPosition = (-1.366234297511774, 1.3236580723856064, 1.7017883885915899, 6.674247054955675, 29.089652104605722, 262.23495534477695, 243.41773973983655, -1941.5159573914977, -766.7305020516261, 24.656845342687614, -0.6149016283510105, 48.45240520806483, 115.4941824677709)
# RMSE = 27.05899365448409


###############################################################################
# Clark GMSL [-2.6 Myr - 0]
###############################################################################

# emcee + dynesty
# StartPosition = (-1, 1, 1, 1, 1, 1, 1, -1200, -800, 1, 1, 1, 1)
# StartPosition = ()
# RMSE = 

# RAMP94 solution
# StartPosition = (-0.1685305244176334, 0.27761644818542663, 2.1543496440725676, 4.62837207530265, 62.34708494726112, 66.77245762968982, 97.54692000195405, -1370.2408631579362, -796.7048662331799, 1, 1, 1, 1)
# StartPosition = (-0.55174878981966, 0.697136833290338, 1.6357664609767206, 4.579523054971209, 1.4076788688084536, 186.43506395494632, -498.98136979405206, -993.0933430364157, -671.8527542202517, 138.563314155537, 1.802625469760802, 58.48309054682487, 54.004530674112914)
# RMSE = 28.984472665410063

# StartPosition = (-0.55174878981966, 0.697136833290338, 1.6357664609767206, 4.579523054971209, 30, 100, 250, -2000, -800, 12, -0.6, 40, 80)
StartPosition = (-1.1343938917360674, 1.257769829808808, 1.7459735984473355, 6.583527120891631, 19.739059337807475, 187.2822360887286, 228.88535071966464, -2205.7333479176727, -762.0781747874983, 25.98771908505603, -0.5157573150308963, 50.907066558484985, 102.93120460213129)
# RMSE = 27.69864387774322


###############################################################################
# Clark GMSL [-3 Myr - 0]
###############################################################################

# 2.6 Myr solution
# StartPosition = (-1.1343938917360674, 1.257769829808808, 1.7459735984473355, 6.583527120891631, 19.739059337807475, 187.2822360887286, 228.88535071966464, -2205.7333479176727, -762.0781747874983, 25.98771908505603, -0.5157573150308963, 50.907066558484985, 102.93120460213129)
# StartPosition = (-1.3014659840990286, 1.3410580284021443, 1.8314459853094935, 6.275509922825426, 20.723172931128165, 125.16886237361462, 253.32800914338256, -2963.6070318180673, -750.5822044104941, 41.48953451369322, -0.6446180201016881, 12.484529249383387, 128.35323809865585)
# RMSE = 28.20743049201896
# StartPosition = (-1.0044228352725213, 1.1886264564724343, 1.8815088426846387, 5.832452149491019, 10.834480132249098, 90.45897876887352, 227.54893987755293, -2792.5280566469587, -783.6206988079153, 37.59163900341491, -0.4284048206986881, 6.985389647858085, 480.8717158200295)
# RMSE = 28.04970421411239


# Parameters = [aEsi, aO, ag, taud, vi, v01, v02, t1, t2, I01, I02, v11, v12]
#                0    1    2   3     4   5    6    7   8   9   10   11    12   

parameter_names = ['aEsi', 'aO', 'ag', 'taud', 'vi', 'v01', 'v02', 't1', 't2', 'I01', 'I02', 'v11', 'v12']



   
# Set resolution of model
resolution = 1000

# Start year in kyr 
start_year = -int(2_600)

# Future simulation years in kyr (between 0 and 1_000)
future_time = int(250)

# Gap included for tuning: Model is not tuned during this time interval. gap=(start_gap[kyr BP], end_gap[kyr BP])
# gap = (-int(1_200), -int(800)) 
# gap = (-int(2_000), -int(600)) 
# gap = (-int(130), -int(0)) 
gap = None


# Set title for Plot
title = 'RAMP-2'

# Set sea-level data: either Berends, Rohling (LR04 based + tuned age), Clark (GMSL from Clark et al., 2025), 
sea_level_data = 'Clark'   # options: 'Berends', 'Rohling', 'Clark'


# save data in binary file with key:value pairs
# save_data = '../Data/RAMP-2_2,6Myr_Berends.pkl'
save_data = None


time_steps = int((-start_year+future_time)*1e3/resolution)  # number of timesteps in model (with steps in future)

print(f"Model resolution: {resolution} years")
print(f"Number of timesteps: {time_steps}")   
print(f"Simulated time interval: {start_year} kyr BP - {future_time} kyr")  
if gap!=None:
    print(f"Gap for tuning: {gap[0]} kyr - {gap[1]} kyr")
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
# Compute the derivative for each time step
def Phi(i, v):
    
    # Orbital forcing
    I = aEsi*Esi[i] + aO*EnO[i]
    
    if S[0] == "g" :
        dvdt = -I + ag
   
    else :
        dvdt = -I - v/taud     
    
    return dvdt


##########################################################
#Compute the modelled volume for the best parameters using the Runge–Kutta 4th order method
def modelledVolume(start_year, future_time, vi, n) :
    v = np.zeros(n+1)
    v[0] = vi
    state.append(S[1])
    
    # full simulation time
    sim_time = abs(start_year)+abs(future_time)
    
    step = (future_time-start_year)/float(n)
    print('Step: ', step)
    for i in range(n):
        # current time t
        t = start_year + (i*sim_time/time_steps)
        
        # Orbital forcing
        I = aEsi*Esi[2*i] + aO*EnO[2*i]
        
        # thresholds for state changes (use Esi and EnO at full time steps only)
        test_threshold_gd = v[i]*I + v[i]
        test_threshold_dg = v[i]*I    
            
        # ---------------------------------------------------
        # RAMP for v0(t)
        
        # t<t1: Before Ramp
        if t < t1:
            v0_t = v01
            
        # t1<=t<=t2: Ramp
        elif t1 <= t <= t2:
            # v0_t = v0 - ((v0-v0_prime)/np.abs(t2-t1)) * np.abs(t-t1) 
            v0_t = v01 + (v02-v01)/(t2-t1) * (t-t1) 
        
        # t2<t: After Ramp
        else:
            v0_t = v02
            
        
        # ---------------------------------------------------
        # RAMP for I0(t)
        
        # t<t1: Before Ramp
        if t < t1:
            I0_t = I01
            
        # t1<=t<=t2: Ramp
        elif t1 <= t <= t2:
            # v0_t = v0 - ((v0-v0_prime)/np.abs(t2-t1)) * np.abs(t-t1) 
            I0_t = I01 + (I02-I01)/(t2-t1) * (t-t1) 
        
        # t2<t: After Ramp
        else:
            I0_t = I02
            
        # ---------------------------------------------------
        # RAMP for v1(t)
        
        # t<t1: Before Ramp
        if t < t1:
            v1_t = v11
            
        # t1<=t<=t2: Ramp
        elif t1 <= t <= t2:
            # v0_t = v0 - ((v0-v0_prime)/np.abs(t2-t1)) * np.abs(t-t1) 
            v1_t = v11 + (v12-v11)/(t2-t1) * (t-t1) 
        
        # t2<t: After Ramp
        else:
            v1_t = v12
            
        
        if S[0] == "g":
            if test_threshold_gd>v0_t: 
                S[0] = "d"
                S[1] = i
                
        else :
            if I<I0_t and v[i]<v1_t:
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



###############################################################################
###############################################################################
###############################################################################


#######################################################
# calculate lower and upper bound for v, where deglaciation/glaciation starts
def calc_bounds(time_steps, v):
    v0s = np.zeros(time_steps+1)
    v0_bounds = np.zeros(time_steps+1)
    v1s = np.zeros(time_steps+1)
    v1_bounds = np.zeros(time_steps+1)
    # v0_bound = v0(t)-insolation (eq.5 from Legrain et.al.)
    
    # complete simulation time
    sim_time = abs(start_year)+abs(future_time)
    print('sim_time: ', sim_time)
    print('time_steps: ', time_steps)
    
    for i in range(time_steps+1):
        # current time t
        t = start_year+(i*sim_time/time_steps)    
        
        # t<t1: Before Ramp
        if t < t1:
            v0_t = v01
            
        # t1<=t<=t2: Ramp
        elif t1 <= t <= t2:
            # v0_t = v0 - ((v0-v0_prime)/np.abs(t2-t1)) * np.abs(t-t1) 
            v0_t = v01 + (v02-v01)/(t2-t1) * (t-t1) 
        
        # t2<t: After Ramp
        else:
            v0_t = v02
            
        # t<t1: Before Ramp
        if t < t1:
            v1_t = v11
            
        # t1<=t<=t2: Ramp
        elif t1 <= t <= t2:
            # v0_t = v0 - ((v0-v0_prime)/np.abs(t2-t1)) * np.abs(t-t1) 
            v1_t = v11 + (v12-v11)/(t2-t1) * (t-t1) 
        
        # t2<t: After Ramp
        else:
            v1_t = v12
            
        
        v0s[i] = v0_t
        v0_bounds[i] = v0s[i] - v[i]*(aEsi*Esi[2*i]+aO*EnO[2*i])
        v1s[i] = v1_t
        v1_bounds[i] = v[i]
    
    time_bounds = np.arange(-start_year*1e3, -future_time*1e3-1, -resolution)*1e-3
    return (time_bounds, v0s, v0_bounds, v1s, v1_bounds)



#####################################################################
# Interpolates given data and time array linearly
# A new time array is created in interval (-start_year,-future_time) with given resolution (in years)
# e.g. [2000, ..., -1000] for interval 2Myr BP - 1Myr in future
def np_interpolation(array, name, resolution, time, start_year=-start_year, future_time=future_time, sea_data='Berends'): 
    if start_year>=3_600 or start_year<0:
        raise ValueError('start_year must be between -3_599 and 0!')
        
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
        
        # default resolution for Rohling and Clark: 1kyr
        elif sea_data=='Rohling' or sea_data=='Clark':
            if resolution==1000:
                print(f'{sea_data} sea-level data: Resolution set to default. Skipping interpolation step!')
                return (time, array)  
            else:
                # create new time array 
                print(f'{sea_data} sea-level data: Resolution interpolated.')
                new_time = np.arange(0, start_year*1e3+1, resolution)*1e-3
                
                new_array = np.interp(new_time, np.flip(time), np.flip(array))
                
                return (np.flip(new_time), np.flip(new_array))
            
        else: 
            raise ValueError("sea_level_data must be either 'Berends', 'Rohling' or 'Clark'!")
    
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


#####################################################################
#####################################################################
# Function for calculating the Bayesian Information Criterion (BIC)
# BIC = -2*LogLikelihood + N_Params*ln(N_DataPoints)                             
# def calc_BIC(parameters, sea, sea_model, n_independent=1):
#     # use only reconstructed sea ice until present, not future predictions
#     sea_model = sea_model[:len(sea):n_independent]
    
#     # Number of (independent) data points
#     sea = sea[::n_independent]
#     N = len(sea)
    
#     # Number of parameters
#     n_params = len(parameters)
    
#     # calculate chis quared
#     chi_squared = np.sum(np.square(sea_model-sea)/sea_model)
    
#     # BIC
#     BIC = -2*np.log(chi_squared)+ n_params*np.log(N)
    
#     return BIC

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

# symmetric mean absolute percentage error (SMAPE)  [%]  
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    

########################################################

#Best parameters
aEsi = StartPosition[0]
aO = StartPosition[1]
ag = StartPosition[2]
taud = StartPosition[3]
vi = StartPosition[4]
v01 = StartPosition[5]
v02 = StartPosition[6]
t1 = StartPosition[7]
t2 = StartPosition[8]
I01 = StartPosition[9]
I02 = StartPosition[10]
v11 = StartPosition[11]
v12 = StartPosition[12]
    
S = ["g",0]
# S = ["d",1]


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

elif sea_level_data=='Clark':
    data_sea = pd.read_excel('../Data/Clark_2025_GMSL.xlsx')
    time_sea = 1e3*data_sea.loc[:, 'Age (Ma)'].to_numpy()
    sea = -1*data_sea.loc[:, 'Sea level (m)'].to_numpy()
    
    mask_interval = np.where(np.logical_and(-time_sea>=start_year, 
                                            -time_sea<=0))
    time_sea = time_sea[mask_interval][::-1]  # reverse array, s.t. it starts from oldest age
    sea = sea[mask_interval][::-1]

else:
    raise ValueError("sea_level_data must be 'Berends', 'Rohling', or 'Clark'!")


# Load orbital data from Laska
data_orbital = np.loadtxt('../Data/Orbital_Params_-3,6MA-2MA_1kyr_steps.txt')
time = data_orbital[:,0]
mask_interval = np.where(np.logical_and(time>=start_year, 
                                        time<=future_time))
time = -1*time[mask_interval]
esinomega = data_orbital[:,1][mask_interval]#[::-1]    # removed -1 from Legrain's initial model
O = data_orbital[:,3][mask_interval]#[::-1]
    

####################################################################
# Interpolate or choose data accordingly to set resolution (Default resolution = Resolution of loaded data = 100 years)
time_sea, sea = np_interpolation(sea, 'sea', resolution, time_sea, sea_data=sea_level_data)  
# sea_std = np_interpolation(sea_std, 'sea', resolution, time_sea, sea_data=sea_level_data)[1]  
# sea_std = np.where(sea_std<1, 1, sea_std)
esinomega = np_interpolation(esinomega, 'esinomega', resolution, time, sea_data=sea_level_data)[1]
time, O = np_interpolation(O, 'O', resolution, time, sea_data=sea_level_data)


#####################################################################
#Normalization of parameters input
Esi = normalise(esinomega, length=len(sea))
EnO = normalise(O, length=len(sea))

#Interpolation to get data at the time step of 500 years (for half-step Runge-Kutta computation)
Esi = interpol(Esi,2)
EnO = interpol(EnO,2)
time_halfsteps = interpol(time,2)


##########################################################
#Modelling of the ice volume for the best parameters fit
icevolume = modelledVolume(start_year, future_time, vi, time_steps)
    
# calculate bounds for deglaciation/glaciation
time_bounds, v0s, v0_bounds, v1s, v1_bounds = calc_bounds(time_steps, icevolume)


#calcul de l'écart modele donnees a chaque pas de temps
residuals = []
sum_residuals = 0
for i in range (len(sea)):
    sum_residuals = sum_residuals + (sea[i]-icevolume[i])**2
    residuals.append(sea[i]-icevolume[i])
    


##########################################################
# Spectral analysis: Periodogram
# Split into 3 diagrams: before, during and after RAMP
# step size in kyr
step = (future_time-start_year)/time_steps

# index to the left of ramp start (t1)
pre_ramp = int(np.floor(-start_year+t1) / step)
print(f'Time of Ramp start: {time[pre_ramp]}')

# index to the left of ramp start (2)
post_ramp = int(np.floor(-start_year+t2) / step)
print(f'Time of Ramp end: {time[post_ramp]}')

ice_pre_ramp = icevolume[:pre_ramp+1]
sea_pre_ramp = sea[:pre_ramp+1]

ice_ramp = icevolume[pre_ramp+1:post_ramp+1]
sea_ramp = sea[pre_ramp+1:post_ramp+1]

ice_post_ramp = icevolume[post_ramp+1:]
sea_post_ramp = sea[post_ramp+1:]


(f_ice_pre_ramp, Power_ice_pre_ramp) = scipy.signal.periodogram(ice_pre_ramp, fs=1/step, scaling='spectrum') 
(f_sea_pre_ramp, Power_sea_pre_ramp) = scipy.signal.periodogram(sea_pre_ramp, fs=1/step, scaling='spectrum')
(f_ice_ramp, Power_ice_ramp) = scipy.signal.periodogram(ice_ramp, fs=1/step, scaling='spectrum') 
(f_sea_ramp, Power_sea_ramp) = scipy.signal.periodogram(sea_ramp, fs=1/step, scaling='spectrum')
(f_ice_post_ramp, Power_ice_post_ramp) = scipy.signal.periodogram(ice_post_ramp, fs=1/step, scaling='spectrum') 
(f_sea_post_ramp, Power_sea_post_ramp) = scipy.signal.periodogram(sea_post_ramp, fs=1/step, scaling='spectrum')

##########################################################
# Spectral analysis: Spectrogram
f_ice_spectrogram, t_ice_spectrogram, spectrogram_ice = scipy.signal.spectrogram(icevolume, fs=1/step, scaling='spectrum')
f_sea_spectrogram, t_sea_spectrogram, spectrogram_sea = scipy.signal.spectrogram(sea, fs=1/step, scaling='spectrum') # , window=('hann', 0.25)

#############################################################################################################################
##########################################################
# Spectral analysis: Periodogram
# Split into 2 diagrams: before,after 800 kyr
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


(f_ice_pre_split, Power_ice_pre_split) = scipy.signal.periodogram(ice_pre_split, fs=1/step, scaling='spectrum') 
(f_sea_pre_split, Power_sea_pre_split) = scipy.signal.periodogram(sea_pre_split, fs=1/step, scaling='spectrum')
(f_ice_post_split, Power_ice_post_split) = scipy.signal.periodogram(ice_post_split, fs=1/step, scaling='spectrum') 
(f_sea_post_split, Power_sea_post_split) = scipy.signal.periodogram(sea_post_split, fs=1/step, scaling='spectrum')


#############################################################################################################################
##########################################################
# Spectral analysis: LombScargle Periodogram 
# Split into 2 diagrams: before and after 800 kyr
# step size in kyr
f_sea_LombScargle_pre_split, P_sea_LombScargle_pre_split = LombScargle(time_sea_pre_split, sea_pre_split, normalization='standard').autopower(minimum_frequency=1/150, maximum_frequency=1/10)
f_sea_LombScargle_post_split, P_sea_LombScargle_post_split = LombScargle(time_sea_post_split, sea_post_split, normalization='standard').autopower(minimum_frequency=1/150, maximum_frequency=1/10)
f_ice_LombScargle_pre_split, P_ice_LombScargle_pre_split = LombScargle(time_sea_pre_split, ice_pre_split, normalization='standard').autopower(minimum_frequency=1/150, maximum_frequency=1/10)
f_ice_LombScargle_post_split, P_ice_LombScargle_post_split = LombScargle(time_sea_post_split, ice_post_split, normalization='standard').autopower(minimum_frequency=1/150, maximum_frequency=1/10)


#############################################################################################################################
#########################################################
# calculate RMSE, MAE, R2, SMAPE and BIC 
# rmse = np.sqrt(np.sum(np.square(sea-icevolume[:len(sea)]))/len(sea))
rmse = root_mean_squared_error(y_true=sea, y_pred=icevolume[:len(sea)])
if gap!=None:
    time_steps_no_future = int((-start_year+0)*1e3/resolution)
    
    residuals_gap = (sea-icevolume[:len(sea)])**2
    # calculate IDs of gap, where to exclude for RMSE
    step = abs(start_year)/time_steps_no_future
    gap_start_id = int(abs(start_year-gap[0])/step)
    gap_end_id = int(abs(start_year-gap[1])/step)
    
    # delete gap interval to be not included in RMSE
    residuals_gap = np.delete(residuals_gap, range(gap_start_id, gap_end_id+1))
    
    gap_rmse = np.sqrt(np.sum(residuals_gap)/len(residuals_gap))
    
mae = mean_absolute_error(y_true=sea, y_pred=icevolume[:len(sea)])
R2 = r2_score(y_true=sea, y_pred=icevolume[:len(sea)])
R_value, p_value = pearsonr(icevolume[:len(sea)], sea)
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
if gap!=None:
    print(f"      RMSE (Gap) = {gap_rmse}")
print(f"      MAE = {mae}")
print(f"      R² = {R2}")
print(f"      R = {R_value}, p = {p_value}")
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
if sea_level_data=='Berends':
    label='Berends'
    ylabel='Ice volume (m sl)'
elif sea_level_data=='Rohling':
    label='Rohling'
    ylabel='Ice volume (m sl)'
elif sea_level_data=='Clark':
    label=r'Clark'
    ylabel='Ice volume (m sl)'
else:
    raise ValueError("Sea level data must be one of: 'Berends', 'Rohling' or 'Clark'!")
     
     
fig, ax1 = plt.subplots(figsize=(20,6))
fig.tight_layout(pad=8)
ax1.plot(time_sea, sea, linestyle="-" , color=bright.blue, label=label)
ax1.plot(time, icevolume, color=bright.purple, label="Model")
if gap!=None:
    ax1.axvspan(xmin=-gap[0], xmax=-gap[1], facecolor=bright.yellow, alpha=0.5, label='Gap')


ax1.plot(time_bounds, v0s, linestyle="--" , color=bright.black, label=r"Deglaciation threshold: $v_0(t)$")
    
plt.vlines(x=-t1, ymin=np.min([np.min(icevolume-0.1*np.abs(icevolume)), np.min(sea-0.1*np.abs(sea))]), 
                 ymax=np.max([np.max(icevolume+0.1*np.abs(icevolume)), np.max(sea+0.1*np.abs(sea))]), 
                 linestyle='-', color=bright.green, label=f'Start of RAMP: t1={int(-t1)} ka')
plt.vlines(x=-t2, ymin=np.min([np.min(icevolume-0.1*np.abs(icevolume)), np.min(sea-0.1*np.abs(sea))]), 
                 ymax=np.max([np.max(icevolume+0.1*np.abs(icevolume)), np.max(sea+0.1*np.abs(sea))]), 
                 linestyle='-', color=bright.green, label=f'End of RAMP: t2={int(-t2)} ka')

ax2 = ax1.twinx()
ax2.plot(time, state, linewidth = 0.8, color=bright.grey, label='Model state')

# plt.xticks([0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000])
# ax2.set_yticks([1,0], ['Interglacial', 'Glacial'], rotation=22.5)
ax2.set_yticks([1,0])
ax1.set_xticks(np.arange(-start_year, -future_time-1, -200))

plt.xlim(-start_year,-future_time)
ax1.set_ylim(np.min([np.min(icevolume-0.1*np.abs(icevolume)), np.min(sea-0.1*np.abs(sea))]), 
             np.max([np.max(icevolume+0.1*np.abs(icevolume)), np.max(sea+0.1*np.abs(sea))]))
# plt.gca().invert_yaxis()
ax1.invert_yaxis()
ax1.set_xlabel("Age (ka)",weight='bold')
ax1.set_ylabel(ylabel,weight='bold')
ax2.set_ylabel('Model state')
fig.legend(ncol=3, loc=8)
if title!=None:
    plt.title(title+f';  RMSE={rmse:.2f} m;  R={R_value:.2f}', fontsize=18)
# plt.savefig('../Data/RAMP-2.png', dpi=500, bbox_inches='tight')
plt.show()



# ###############################################################################
# # 2ND PLOT: Periodogram
# fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(12,10), dpi=300)
# fig.tight_layout(pad=5.0)


# ax[0].plot(f_sea_pre_ramp, Power_sea_pre_ramp, linestyle='--', color=bright.blue, label="Berends")
# ax[0].plot(f_ice_pre_ramp, Power_ice_pre_ramp, color=bright.purple, label="Model")
# # mark obliquity and precession cycles + 100 kyr
# ax[0].vlines([1/100, 1/41, 1/23], 0, np.max([np.max(Power_sea_pre_ramp), np.max(Power_ice_pre_ramp)])+10, color='grey', linestyle='--')
# ax[0].set_ylim(0,np.max([np.max(Power_sea_pre_ramp), np.max(Power_ice_pre_ramp)])+10)
# ax[0].set_xlim([1/500,0.05])

# ax[1].plot(f_sea_ramp, Power_sea_ramp, linestyle='--', color=bright.blue, label="Berends")
# ax[1].plot(f_ice_ramp, Power_ice_ramp, color=bright.purple, label="Model")
# # mark obliquity and precession cycles + 100 kyr
# ax[1].vlines([1/100, 1/41, 1/23], 0, np.max([np.max(Power_sea_ramp), np.max(Power_ice_ramp)])+10, color='grey', linestyle='--')
# ax[1].set_ylim(0,np.max([np.max(Power_sea_ramp), np.max(Power_ice_ramp)])+10)
# ax[1].set_xlim([1/500,0.05])

# ax[2].plot(f_sea_post_ramp, Power_sea_post_ramp, linestyle='--', color=bright.blue, label="Berends")
# ax[2].plot(f_ice_post_ramp, Power_ice_post_ramp, color=bright.purple, label="Model")
# # mark obliquity and precession cycles + 100 kyr
# ax[2].vlines([1/100, 1/41, 1/23], 0, np.max([np.max(Power_sea_post_ramp), np.max(Power_ice_post_ramp)])+10, color='grey', linestyle='--')
# ax[2].set_ylim(0,np.max([np.max(Power_sea_post_ramp), np.max(Power_ice_post_ramp)])+10)
# ax[2].set_xlim([1/500,0.05])


# ax[2].set_xlabel("Frequency [1/kyr]", weight='bold')
# ax[1].set_ylabel(r"Squared magnitude spectrum [$m^2$]",weight='bold')


# # ax[0].plot(1/f_sea_pre_ramp, Power_sea_pre_ramp, linestyle='--', color=bright.blue, label="Berends")
# # ax[0].plot(1/f_ice_pre_ramp, Power_ice_pre_ramp, color=bright.purple, label="Model")
# # ax[1].plot(1/f_sea_ramp, Power_sea_ramp, linestyle='--', color=bright.blue, label="Berends")
# # ax[1].plot(1/f_ice_ramp, Power_ice_ramp, color=bright.purple, label="Model")
# # ax[2].plot(1/f_sea_post_ramp, Power_sea_post_ramp, linestyle='--', color=bright.blue, label="Berends")
# # ax[2].plot(1/f_ice_post_ramp, Power_ice_post_ramp, color=bright.purple, label="Model")
# # ax[2].set_xlabel("Periodicity [kyr]", weight='bold')
# # ax[0].set_xlim([0,200])
# # ax[1].set_xlim([0,200])
# # ax[2].set_xlim([0,200])


# for i in range(3):
#     ax[i].text(0.09, 0.8, '100 kyr', color='grey', rotation=0, transform=ax[i].transAxes)
#     ax[i].text(0.4, 0.8, '41 kyr', color='grey', rotation=0, transform=ax[i].transAxes)
#     ax[i].text(0.8, 0.8, '23 kyr', color='grey', rotation=0, transform=ax[i].transAxes)

# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# ax[0].set_title(f'Pre RAMP ({-start_year*1e-3:.1f}-{time[pre_ramp]*1e-3:.1f} Myr BP)')
# ax[1].set_title(f'During RAMP ({time[pre_ramp]*1e-3:.1f} - {time[post_ramp]*1e-3:.1f} Myr BP)')
# ax[2].set_title(f'Post MPT ({time[post_ramp]*1e-3:.1f}-0 Myr BP)')

# # if title!=None:
# #     plt.title(title, fontsize=25)
# # plt.savefig('../Data/RAMP-2_periodogram.png', dpi=300)
# plt.show()


# ###############################################################################
# # 2ND PLOT: Periodogram (Alternative with only 2 subplots)
# fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(12,6), dpi=300)
# fig.tight_layout(pad=5.0)


# ax[0].plot(f_sea_pre_split, Power_sea_pre_split, linestyle='--', color=bright.blue, label="Berends")
# ax[0].plot(f_ice_pre_split, Power_ice_pre_split, color=bright.purple, label="Model")
# # mark obliquity and precession cycles + 100 kyr
# ax[0].vlines([1/100, 1/41, 1/23], 0, np.max([np.max(Power_sea_pre_split), np.max(Power_ice_pre_split)])+10, color='grey', linestyle='--')
# ax[0].set_ylim(0,np.max([np.max(Power_sea_pre_split), np.max(Power_ice_pre_split)])+10)
# ax[0].set_xlim([1/500,0.05])

# ax[1].plot(f_sea_post_split, Power_sea_post_split, linestyle='--', color=bright.blue, label="Berends")
# ax[1].plot(f_ice_post_split, Power_ice_post_split, color=bright.purple, label="Model")
# # mark obliquity and precession cycles + 100 kyr
# ax[1].vlines([1/100, 1/41, 1/23], 0, np.max([np.max(Power_sea_post_split), np.max(Power_ice_post_split)])+10, color='grey', linestyle='--')
# ax[1].set_ylim(0,np.max([np.max(Power_sea_post_split), np.max(Power_ice_post_split)])+10)
# ax[1].set_xlim([1/500,0.05])


# ax[1].set_xlabel("Frequency [1/kyr]", weight='bold')
# # Add a common y-axis label
# fig.text(0.04, 0.5, r"Squared magnitude spectrum [$m^2$]", weight='bold', va='center', rotation='vertical')



# for i in range(2):
#     ax[i].text(0.09, 0.8, '100 kyr', color='grey', rotation=0, transform=ax[i].transAxes)
#     ax[i].text(0.4, 0.8, '41 kyr', color='grey', rotation=0, transform=ax[i].transAxes)
#     ax[i].text(0.8, 0.8, '23 kyr', color='grey', rotation=0, transform=ax[i].transAxes)

# ax[0].legend()
# ax[1].legend()
# ax[0].set_title(f'{-start_year*1e-3:.1f}-{time[pre_split]*1e-3:.1f} Myr BP')
# ax[1].set_title(f'{time[pre_split]*1e-3:.1f}-0 Myr BP')

# # if title!=None:
# #     plt.title(title, fontsize=25)
# # plt.savefig('../Data/RAMP-2_periodogram.png', dpi=300)
# plt.show()


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
# plt.savefig('../Data/RAMP-2_periodogram.png', dpi=300)
plt.show()


###############################################################################
# 3RD PLOT: Pyleoclim scalogram

# IEVOLUME
ts_icevolume = pyleo.Series(time=time, value=icevolume, 
                            time_name='Age', time_unit='ka', 
                            value_name='Icevolume', value_unit='m Sl', 
                            verbose=False)
psd_icevolume = ts_icevolume.spectral()
scal_sig_ar1asym_icevolume = ts_icevolume.wavelet(freq_kwargs={'fmin':1/500,'fmax':1/10,'nf':50}).signif_test(method='ar1asym')

# fig, ax = ts_icevolume.summary_plot(psd_icevolume.beta_est(), scal_sig_ar1asym_icevolume, figsize=(20,8))
fig, ax = scal_sig_ar1asym_icevolume.plot(figsize=(20,8))
plt.xticks(np.hstack([-future_time, np.arange(0, -start_year+1, 200)]))
plt.hlines(y=[23, 41, 100], xmin=-future_time, xmax=-start_year, colors='red', linestyles='--', label='Orbital frequencies')
plt.vlines(0, 10, 500, colors='grey', linestyles='--', linewidth=3)
plt.ylabel('Period (ka)')
plt.show()


# # SEA DATA
# ts_sea = pyleo.Series(time=time_sea, value=sea, 
#                       time_name='Age', time_unit='ka', 
#                       value_name='Target data', verbose=False)
# psd_sea= ts_sea.spectral()
# scal_sig_ar1asym_sea= ts_sea.wavelet(freq_kwargs={'fmin':1/500,'fmax':1/10,'nf':50}).signif_test(method='ar1asym')

# # COHERENCE PLOT
# coh = ts_icevolume.wavelet_coherence(ts_sea, freq_kwargs={'fmin':1/500,'fmax':1/10,'nf':50})
# coh_sig = coh.signif_test()
# coh_sig.dashboard()
# plt.show()

# # ###############################################################################
# # # 3Rd PLOT: Insolation
# Esi = np.array(Esi[::2])
# EnO = np.array(EnO[::2])
# I = aEsi*Esi + aO*EnO
# threshold = icevolume*I

# plt.figure(figsize=(20,6))
# plt.plot(time, I, label='I')
# plt.plot(time, threshold, label='Threshold')
# plt.plot(time_bounds, v0_bounds, label='v0 bounds')
# plt.plot(time_bounds, v0s, label='v0s')
# plt.plot(time_bounds, v1s, label='v1s')
# plt.legend()
# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# plt.show()

# # ###############################################################################
# # 4th PLOT: Precession and Obliquity
# fig, ax1 = plt.subplots(figsize=(20,6))
# plt.plot(time[-300:-250], Esi[-300:-250], color='blue', label='Esi')
# ax1.legend(loc=1)
# ax1.invert_xaxis()

# ax2 = plt.twinx(ax1)
# ax2.plot(time[-300:-250], EnO[-300:-250], color='orange', label='Obliquity')
# ax2.legend()
# ax2.invert_xaxis()
# ax2.hlines(y=0, xmin=0, xmax=50, linestyle='--', color='grey')

# plt.xlim(0, 50)
# plt.show()


# ###############################################################################
# # 3Rd PLOT: Spectrogram
# fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(10,8), dpi=300)
# fig.tight_layout(pad=5.0)
# ax[0].pcolormesh(t_sea_spectrogram, f_sea_spectrogram, spectrogram_sea, shading='gouraud') 
# ax[1].pcolormesh(t_ice_spectrogram, f_ice_spectrogram, spectrogram_ice, shading='gouraud')

# ax[0].set_ylabel('Frequency [1/kyr]')
# ax[1].set_ylabel('Frequency [1/kyr]')
# ax[1].set_xlabel('Years BP [kyr]')

# ax[0].set_ylim([0,0.05])
# ax[1].set_ylim([0,0.05])

# plt.setp(ax, xticks=[1600, 1400, 1200, 1000, 800, 600, 400, 200], xticklabels=['400', '600', '800', '1000', '1200', '1400', '1600', '1800'])

# ax[0].set_title('Berends Spectrogram')
# ax[1].set_title('Model Spectrogram')

# if title!=None:
#     plt.title(title, fontsize=25)
# # plt.savefig('../Data/RAMP-2_spectrogram.png', dpi=300)
# plt.show()


###############################################################################
# # 4TH PLOT: Comparison model-data for the best fit StartPosition
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
# plt.savefig(f'../Data/test2.png', dpi=300)
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


# ###############################################################################
# # 5th Plot: residuals (model-data) for the best fit StartPosition
# boundary = 20

# plt.figure(figsize=(20,4), dpi=300)
# plt.plot(time_sea, residuals, color=bright.green, label="Model")
# plt.hlines(0, -start_year, -future_time-1, color='grey')
# plt.hlines(boundary, -start_year, 0, linestyle='--', color='grey')
# plt.hlines(-boundary, -start_year, 0, linestyle='--', color='grey')

# plt.fill_between(time_sea, residuals, boundary, where=(np.array(residuals) >= boundary), color='red', alpha=0.3)
# plt.fill_between(time_sea, residuals, -boundary, where=(np.array(residuals) <= -boundary), color='blue', alpha=0.3)

# plt.xticks(np.arange(-start_year, -1, -200))
# plt.yticks([boundary, 0, -boundary])


# plt.xlim(-start_year, 0)
# # plt.ylim(-50, 50)
# # plt.gca().invert_xaxis()
# # plt.gca().invert_yaxis()

# plt.xlabel("Age (ka)",weight='bold')
# plt.ylabel('Residuals (model-data)(msl)',weight='bold')
# plt.show()

###############################################################################
# Save data to binary file
if save_data!=None:
    # store all relevant data for later plotting
    data = {'rmse': rmse, 
            'smape': SMAPE, 
            'R2': R2,
            'R': R_value,
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
            'v1s': v1s, 
            'v1_bounds': v1_bounds, 
            'time_bounds': time_bounds,
            'StartPosition': StartPosition, 
            'v01': v01, 
            't1': t1, 
            't2': t2, 
            'taud': taud, 
            'v02': v02, 
            'state': state, 
            'f_ice_pre_ramp': f_ice_pre_ramp,
            'f_sea_pre_ramp': f_sea_pre_ramp, 
            'f_ice_pre_split': f_ice_pre_split,
            'f_sea_pre_split': f_sea_pre_split, 
            'f_ice_ramp': f_ice_ramp, 
            'f_sea_ramp': f_sea_ramp, 
            'f_ice_post_ramp': f_ice_post_ramp, 
            'f_sea_post_ramp': f_sea_post_ramp, 
            'f_ice_post_split': f_ice_post_split, 
            'f_sea_post_split': f_sea_post_split, 
            'Power_ice_pre_ramp': Power_ice_pre_ramp,
            'Power_sea_pre_ramp': Power_sea_pre_ramp, 
            'Power_ice_pre_split': Power_ice_pre_split,
            'Power_sea_pre_split': Power_sea_pre_split, 
            'Power_ice_ramp': Power_ice_ramp, 
            'Power_sea_ramp': Power_sea_ramp, 
            'Power_ice_post_ramp': Power_ice_post_ramp, 
            'Power_sea_post_ramp': Power_sea_post_ramp,
            'Power_ice_post_split': Power_ice_post_split, 
            'Power_sea_post_split': Power_sea_post_split,
            'pre_ramp': pre_ramp, 
            'pre_split': pre_split, 
            'post_ramp': post_ramp, 
            'time_sea_pre_split': time_sea_pre_split, 
            'time_sea_post_split': time_sea_post_split, 
            'f_sea_LombScargle_pre_split': f_sea_LombScargle_pre_split,
            'P_sea_LombScargle_pre_split': P_sea_LombScargle_pre_split, 
            'f_sea_LombScargle_post_split': f_sea_LombScargle_post_split, 
            'P_sea_LombScargle_post_split': P_sea_LombScargle_post_split, 
            'f_ice_LombScargle_pre_split': f_ice_LombScargle_pre_split, 
            'P_ice_LombScargle_pre_split': P_ice_LombScargle_pre_split, 
            'f_ice_LombScargle_post_split': f_ice_LombScargle_post_split, 
            'P_ice_LombScargle_post_split': P_ice_LombScargle_post_split, 
            'params': dict(zip(parameter_names,StartPosition)), 
            'Esi': Esi, 
            'EnO': EnO,
            'esinomega': esinomega,
            'O': O
            }
    
    if gap!=None:
        data['gap'] = gap
    
    # Write to binary file
    with open(save_data, 'wb') as f:
        pickle.dump(data, f)
