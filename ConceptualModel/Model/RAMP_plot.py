#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 09:33:34 2025

@author: pollakf

Note: - Same as RAMP80_jax_plot.py, but changed t1 and t2 into physical units -> negative now + deleted benthic targets
      - changed names/parametrizations to revised manuscript
      - dropped -1* factor when loading esinw data
      -> this was used in the Legrain et al. model, but it changes sign of the actual esinw data
      -> new threshold: v*I +v >v0 & v*I > v1
      - No longer Ik and Ialpha, just one forcing I(t)
      - I(t) = aEsi*Esi + aO*O
      - no alpha_d
      - taud is constant in time
      - co-precession dropped
      - Added Clark GMSL
      - Bounds: params: +/- 10_000
      - 10 params
      
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

# 2.6 Ma solution
# StartPosition = (-0.5361307492885317, 0.45867720689284397, 0.7691545154076493, 6.919653249321868, 125.11325675182707, 12.655055630582524, 27.111071617329003, -0.34492012852899734, -2290.7633271657164, -546.0857039725839)
# StartPosition = (-0.6105814082584402, 0.5286728863924282, 0.8183912177213415, 7.324215911683723, 147.18568525694707, 7.7129836806821, 23.15672861184396, 47.95502689121088, -1960.3541094624434, -54.78903584274841)
# RMSE = 12.848860862701887

# emcee + dynesty
# StartPosition = (-1, 1, 1, 1, 1, 1, 1, 1, 1200, 800)
# StartPosition = (-0.6356483171099399, 0.48453867026034914, 0.8099897084155145, 7.2036984078363275, 140.64093385870865, 6.801738450228356, 21.519710850501042, 52.12031437369467, -1993.0656208112694, -126.82937397402489)
# RMSE = 12.794855126813353

###############################################################################
# Berends [-2.6 Myr - 0]
###############################################################################

# old RAMP77
# StartPosition = (-0.6343723326618866, 0.5182934711183897, 0.8445883819764034, 7.030270525278787, 150.5718181816133, 0.3377298036986667, 26.45404982306816, -0.431684610132558, -2417.0932080298658, -273.15206028168217)
StartPosition = (-0.5361307492885317, 0.45867720689284397, 0.7691545154076493, 6.919653249321868, 125.11325675182707, 12.655055630582524, 27.111071617329003, -0.34492012852899734, -2290.7633271657164, -546.0857039725839)
# RMSE = 12.29042588916936

# emcee + dynesty
# StartPosition = (-1, 1, 1, 1, 1, 1, 1, 1, 1200, 800)
# StartPosition = (-0.6289310570817761, 0.5169311919688128, 0.7837026934808439, 6.33539400864169, 135.63528474493987, 13.625405211968726, 22.315865762137435, -2.3270945066761213, -2308.561219187777, -598.860239137025)
# RMSE = 12.344179552421743

# GAP (1.2-0.8)
# StartPosition = (-1, 1, 1, 1, 1, 1, 1, 1, 1200, 800)
# StartPosition = (-0.5506538412424788, 0.4602691945414108, 0.7685889607415675, 6.643603226498271, 123.23109730629506, 11.802818654321982, 27.149880070089466, -0.764080903452907, -2228.295389043347, -966.0617048934067)
# RMSE = 12.597055158045194
# RMSE (Gap) = 11.866534063567938

# GAP (2-0.6)
# StartPosition = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
# StartPosition = (-0.5941965476628766, 0.5577967336410011, 0.7917072264931624, 6.568277218636194, 143.31626200734192, 14.115109912454166, 24.863774673370727, -2.00490953360761, -2338.994564092796, -156.760073789062)
# RMSE = 14.289224775488968
# RMSE (Gap) = 11.914917011710223

###############################################################################
# Berends [-3 Myr - 0]
###############################################################################

# 2.6 Ma solution
# StartPosition = (-0.5361307492885317, 0.45867720689284397, 0.7691545154076493, 6.919653249321868, 125.11325675182707, 12.655055630582524, 27.111071617329003, -0.34492012852899734, -2290.7633271657164, -546.0857039725839)
# StartPosition = (-0.5551212142640435, 0.41108405955682137, 0.7755188386961052, 7.577490602000476, 124.13342501983176, 9.9258603899663, 12.367493653321386, -0.8230655838153481, -2333.9265331910105, -573.3087864574384)
# RMSE = 12.041442297143435

###############################################################################
# Rohling [-2.6 Myr - 0]
###############################################################################

# Berends solution
# StartPosition = (-0.5361307492885317, 0.45867720689284397, 0.7691545154076493, 6.919653249321868, 125.11325675182707, 12.655055630582524, 27.111071617329003, -0.34492012852899734, -2290.7633271657164, -546.0857039725839)
# StartPosition = (-0.5753596967855401, 0.5397987524648897, 0.7735860962110337, 7.1770424566808515, 144.64490149064397, 17.05617285838605, 27.578641281094747, -0.8701786676130041, -2595.8443618418632, -251.90083845744735)
# RMSE = 12.737232897311708

# emcee + dynesty
# StartPosition = (-1, 1, 1, 1, 1, 1, 1, 1, -1200, -800)
# StartPosition = (-0.5753483806263375, 0.5397933840002906, 0.7735592032764771, 7.177211418084994, 144.5786070074397, 17.05620499395536, 27.58549276555084, -0.4864847533651755, -2589.6902046043797, -252.87493980898336)
# RMSE = 12.737248134545712


# Parameters = [aEsi, aO, ag, taud, v02, v1, vi, v01, t1, t2]
#                0    1    2   3     4    5   6   7    8   9  

parameter_names = ['aEsi', 'aO', 'ag', 'taud', 'v02', 'v1', 'vi', 'v01', 't1', 't2']

   
# Set resolution of model
resolution = 1000

# Start year in kyr 
start_year = -int(2_600)

# Future simulation years in kyr (between 0 and 1_000)
future_time = int(250)

# Gap included for tuning: Model is not tuned during this time interval. gap=(start_gap[kyr BP], end_gap[kyr BP])
# gap = (-int(2_000), -int(600))
# gap = (-int(1_200), -int(800)) 
# gap = (-int(2_000), -int(600)) 
# gap = (-int(130), -int(0)) 
gap = None


# Set title for Plot
title = 'RAMP'

# Set sea-level data
sea_level_data = 'Berends'   # options: 'Berends', 'Rohling', or 'Clark'


# save data in binary file with key:value pairs
# save_data = '../Data/RAMP_2,6Myr_Berends_Gap-2000-600.pkl'
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
        
        # t < t1: Before Ramp
        if t < t1:
            v0_t = v01
            
        # t1 <= t <= t2: During Ramp
        elif t1 <= t <= t2:
            v0_t = v01 + (v02-v01)/(t2-t1) * (t-t1) 
        
        # t2 < t: After Ramp
        else:
            v0_t = v02
        
        if S[0] == "g":
            if test_threshold_gd>v0_t and test_threshold_dg>v1:
                S[0] = "d"
                S[1] = i
                
        else :
            if test_threshold_dg<v1 and test_threshold_gd<v0_t:
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
        t = start_year + (i*sim_time/time_steps)    
        
        # t < t1: Before Ramp
        if t < t1:
            v0_t = v01
            
        # t1 <= t <= t2: Ramp
        elif t1 <= t <= t2:
            v0_t = v01 + (v02-v01)/(t2-t1) * (t-t1) 
        
        # t2 < t: After Ramp
        else:
            v0_t = v02
        
        v0s[i] = v0_t
        v0_bounds[i] = v0s[i] - v[i]*(aEsi*Esi[2*i]+aO*EnO[2*i])
        v1s[i] = v1
        v1_bounds[i] = v[i]*(aEsi*Esi[2*i]+aO*EnO[2*i])
    
    time_bounds = np.arange(-start_year*1e3, -future_time*1e3-1, -resolution)*1e-3
    return (time_bounds, v0s, v0_bounds, v1s, v1_bounds)



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
v02 = StartPosition[4]
v1 = StartPosition[5]
vi = StartPosition[6]
v01 = StartPosition[7]
t1 = StartPosition[8]
t2 = StartPosition[9]
    
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
    label='Clark'
    ylabel='Ice volume (m sl)'
else:
    raise ValueError("Sea level data must be one of: 'Berends', 'Rohling', or 'Clark'!")
     
     
fig, ax1 = plt.subplots(figsize=(20,6))
fig.tight_layout(pad=8)
ax1.plot(time_sea, sea, linestyle="-" , color=bright.blue, label=label)
ax1.plot(time, icevolume, color=bright.purple, label="Model")
if gap!=None:
    ax1.axvspan(xmin=-gap[0], xmax=-gap[1], facecolor=bright.yellow, alpha=0.5, label='Gap')


ax1.plot(time_bounds, v0s, linestyle="--" , color=bright.black, label=r"Deglaciation threshold: $v_0(t)$")
    
plt.vlines(x=-t1, ymin=np.min([np.min(icevolume-0.1*np.abs(icevolume)), np.min(sea-0.1*np.abs(sea))]), 
                 ymax=np.max([np.max(icevolume+0.1*np.abs(icevolume)), np.max(sea+0.1*np.abs(sea))]), 
                 linestyle='-', color=bright.green, label=f'Start of RAMP: t1={int(t1)} kyr')
plt.vlines(x=-t2, ymin=np.min([np.min(icevolume-0.1*np.abs(icevolume)), np.min(sea-0.1*np.abs(sea))]), 
                 ymax=np.max([np.max(icevolume+0.1*np.abs(icevolume)), np.max(sea+0.1*np.abs(sea))]), 
                 linestyle='-', color=bright.green, label=f'End of RAMP: t2={int(t2)} kyr')

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
# # plt.savefig('../Data/RAMP_periodogram.png', dpi=300)
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
# plt.savefig('../Data/RAMP_periodogram.png', dpi=300)
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

# ###############################################################################
# # 3Rd PLOT: Insolation
Esi = np.array(Esi[::2])
EnO = np.array(EnO[::2])
I = aEsi*Esi + aO*EnO
threshold = icevolume*I

plt.figure(figsize=(20,6))
plt.plot(time, I, label='I')
plt.plot(time, threshold, label='Threshold')
plt.plot(time_bounds, v0_bounds, label='v0 bounds')
plt.plot(time_bounds, v0s, label='v0s')
plt.plot(time_bounds, v1s, label='v1s')
plt.legend()
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.show()

# ###############################################################################
# 4th PLOT: Precession and Obliquity
fig, ax1 = plt.subplots(figsize=(20,6))
plt.plot(time[-300:-250], Esi[-300:-250], color='blue', label='Esi')
ax1.legend(loc=1)
ax1.invert_xaxis()

ax2 = plt.twinx(ax1)
ax2.plot(time[-300:-250], EnO[-300:-250], color='orange', label='Obliquity')
ax2.legend()
ax2.invert_xaxis()
ax2.hlines(y=0, xmin=0, xmax=50, linestyle='--', color='grey')

plt.xlim(0, 50)
plt.show()


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
# # plt.savefig('../Data/RAMP_spectrogram.png', dpi=300)
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
# if sea_level_data=='Clark-d18Osw':
#     boundary = 0.1

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
            'v1': v1,
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
