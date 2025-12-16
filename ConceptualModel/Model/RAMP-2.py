#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 10:07:26 2025

@author: pollakf

Note: - Same as RAMP97_jax.py, but changed time according to Ganopolski
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
      
Note2: JAX uses by default float32. float64 is much slower and has to be 
       explicitly enabled
       
"""               


import numpy as np 
import emcee
from multiprocessing import Pool
import time as time_module 
from MyModules import ptemcee_modified as ptm
from numba import njit
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import pandas as pd
import dynesty
from dynesty import plotting as dyplot
from tqdm import tqdm
import os
import scipy
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=20'
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
from jax import jit
from pytensor.graph import Apply, Op
import pytensor.tensor as pt
import pymc as pm
import arviz as az

# enabling doubleprecision in JAX
# jax.config.update("jax_enable_x64", True)

# install tqdm package to see progress bar 
# conda install -c conda-forge tqdm


# for walkers in [100]:
#     print('\n=======================================\n')
#     print('\n=======================================\n')
#     print(f'\nNumbers of walkers used: {walkers}\n')
#     print('\n=======================================\n')
#     print('\n=======================================\n')
#     for ncores in [1]:
#         print('\n=======================================\n')
#         print(f'\nNumbers of cores used: {ncores}\n')
#         print(f'\nNumbers of walkers used: {walkers}\n')
#         for i in range(3):
            
    
tic = time_module.perf_counter() 

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

# 2.6 Myr solution
# StartPosition = (-0.5884169134374966, 0.5318247545292409, 0.8217986175738236, 7.194732231231515, 38.504040523227296, 42.107975558236696, 142.3857878945521, -2272.7141119990124, -31.239793423460917, 1.613503486041131, 0.11592898666437554, 0.584480971205182, 47.9581056570014)
# StartPosition = (-0.5906552776301797, 0.5024707574927784, 0.7961382269542748, 8.02793599456681, 9.839366389736236, 15.757935850100315, 135.7744316147947, -2992.2050161664556, -62.71772798023692, 2.239371275475545, 0.003837812858366485, -10.360393440389766, 43.87382534302778)
# RMSE = 11.656751863427248


###############################################################################
# Clark GMSL [-2 Myr - 0]
###############################################################################

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


#Number of walkers (verifying ; nwalkers > 2 * number of parameters)
nwalkers = 50

#Number of iterations 
niterations = 100_000   #100_000   (for 1000yr res: 500_000 + walker_jumps=0.3)

#Define the first position of each walkers relatively to StartPosition. When walkers_jump is high, walkers are far from StartPosition. 
walkers_jump = 0.3

# Number of live points (only needed for dynesty sampler)
nlive = 1_024 #10*1024 #1_024

# Number of temperatures (only for parallel tempering)
ntemps = 20

# Set resolution of model in yr
resolution = 1000

# Start year in kyr 
start_year = -int(2_600)

# Do not change (ends simulation at present time 0) 
future_time = int(0)

# Gap included for tuning: Model is not tuned during this time interval. gap=(start_gap[kyr BP], end_gap[kyr BP])
# gap = (-int(1_200), -int(800)) 
# gap = (-int(2_000), -int(600)) 
# gap = (-int(130), -int(0)) 
# gap = (-int(2600), -int(700))
gap = None

time_steps = int(-start_year*1e3/resolution)  # number of timesteps in model

# Select tuning procedure
tuning = 'emcee'  # Options: 'emcee', 'ptemcee', 'pymc', 'dynesty'
dynesty_static = True

# Set sea-level data: either Berends, Rohling (LR04 based + tuned age), Clark (GMSL from Clark et al., 2025), 
sea_level_data = 'Clark'   # options: 'Berends', 'Rohling', 'Clark'


blob = False
if tuning!='emcee':
    blob = False

print(f"Model resolution: {resolution} years")
print(f"Number of timesteps: {time_steps}")   
print(f"Number of iterations: {niterations}") 
print(f"Simulated time interval: {start_year} kyr BP - present")
if gap!=None:
    print(f"Gap for tuning: {gap[0]} kyr - {gap[1]} kyr") 
print(f"Tuning library: {tuning}")
print(f"Sea-level data from {sea_level_data} et al.\n")

###################################################################
# Normalise a distribution    
@njit
def normalise(sample):
    m = np.mean(sample)
    std = np.std(sample)
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



#####################################################################
# Define the Phi function
def Phi(i, vt, params, state, global_vars, sim_time, time_steps):
    Esi = global_vars[0]
    EnO = global_vars[1]
    
    # Orbital forcing
    I = params[0] * Esi[i] + params[1] * EnO[i]
    
    # Calculate change in ice volume dv/dt
    dvdt = jnp.where(state == 0,
                     -I + params[2],
                     -I - vt/params[3])
    
    return dvdt


# Compute the modelled volume for a set of input parameters using the Runge–Kutta 4th order method
def modelledVolume(params, global_vars):
    vi = params[4]
    state = 0  # 'g' state represented as 0
    # state = 1  # 'd' state represented as 0
    
    params = jnp.array(params)
    global_vars = jnp.array(global_vars)
    Esi = global_vars[0]
    EnO = global_vars[1]
    
    # total simulation time
    sim_time = jnp.abs(start_year) 
    
    vt = jnp.zeros(time_steps + 1)
    vt = vt.at[0].set(vi)
    step = -start_year/float(time_steps)
    
    def body(i, val):
        vt, state = val
        
        # current time t (full timesteps only)
        t = start_year + (i * sim_time / time_steps)
        
        # Orbital forcing
        I = params[0] * Esi[2*i] + params[1] * EnO[2*i]
        
        # thresholds for state changes (use Esi, EnO at full time steps only)
        test_threshold_gd = vt[i]*I + vt[i]

        # ----------------------------------------------
        # RAMP for v0(t)
        
        # t<t2: Before Ramp
        def before_ramp():
            v0_t = params[5]
            return v0_t

        # t2 <= t <= t1: Ramp
        def during_ramp():
            v0_t = params[5] + ((params[6] - params[5]) / (params[8] - params[7])) * (t - params[7])
            return v0_t

        # t1 < t: After Ramp
        def after_ramp():
            v0_t = params[6]
            return v0_t

        v0_t = jax.lax.cond(t < params[7], before_ramp, 
                                   lambda: jax.lax.cond(t<=params[8] , during_ramp, after_ramp))
        
        # ----------------------------------------------
        # RAMP for I0(t)
        
        # t<t2: Before Ramp
        def before_ramp():
            I0_t = params[9]
            return I0_t

        # t2 <= t <= t1: Ramp
        def during_ramp():
            I0_t = params[9] + ((params[10] - params[9]) / (params[8] - params[7])) * (t - params[7])
            return I0_t

        # t1 < t: After Ramp
        def after_ramp():
            I0_t = params[10]
            return I0_t

        I0_t = jax.lax.cond(t < params[7], before_ramp, 
                                   lambda: jax.lax.cond(t<=params[8] , during_ramp, after_ramp))
        
        # ----------------------------------------------
        # RAMP for v1(t)
        
        # t<t2: Before Ramp
        def before_ramp():
            v1_t = params[11]
            return v1_t

        # t2 <= t <= t1: Ramp
        def during_ramp():
            v1_t = params[11] + ((params[12] - params[11]) / (params[8] - params[7])) * (t - params[7])
            return v1_t

        # t1 < t: After Ramp
        def after_ramp():
            v1_t = params[12]
            return v1_t

        v1_t = jax.lax.cond(t < params[7], before_ramp, 
                                   lambda: jax.lax.cond(t<=params[8] , during_ramp, after_ramp))

        # check if transition in state
        def check_glacial():
            return jnp.where(test_threshold_gd > v0_t, 1, state)

        def check_deglacial():
            return jnp.where((I < I0_t) & (vt[i] < v1_t), 0, state)

        state = jax.lax.cond(state == 0, check_glacial, check_deglacial)
        
        k1 = Phi(2 * i, vt[i], params, state, global_vars, sim_time, time_steps)
        k2 = Phi(2 * i + 1, vt[i] + k1 * step / 2., params, state, global_vars, sim_time, time_steps)
        k3 = Phi(2 * i + 1, vt[i] + k2 * step / 2., params, state, global_vars, sim_time, time_steps)
        k4 = Phi(2 * i + 2, vt[i] + step * k3, params, state, global_vars, sim_time, time_steps)
        
        vt = vt.at[i + 1].set(vt[i] + step / 6. * (k1 + 2 * k2 + 2 * k3 + k4))
        vt = vt.at[i + 1].set(jnp.where(jnp.isnan(vt[i + 1]), -jnp.inf, vt[i + 1]))
        
        return (vt, state)
    
    vt, state = jax.lax.fori_loop(lower=0, upper=time_steps, body_fun=body, init_val=(vt, state))
    return vt
    

jit_modelledVolume = jit(modelledVolume, backend='cpu')



###############################################################################
###############################################################################
###############################################################################



#####################################################################
#Compute the residuals between model and data for a set of input parameters 
def cost_function_negative(parameters, sea_std):
    sea_model = jit_modelledVolume(parameters, global_vars) 
    residuals = np.square((sea_model-sea)/sea_std)
    
    # if gap exists, exclude from likelihood
    if gap!=None:
        # calculate IDs of gap, when to exclude sea level data for tuning
        step = abs(start_year)/time_steps
        gap_start_id = int(abs(start_year-gap[0])/step)
        gap_end_id = int(abs(start_year-gap[1])/step)
        
        # delete gap interval to be not included in tuning
        residuals = np.delete(residuals, range(gap_start_id, gap_end_id+1))
        
    loglikelihood = -0.5 * np.sum(residuals)
    
    if blob:
        if not np.all(np.isfinite(sea_model)):
            return -np.inf, None
        
        # caluclate RMSE and return as blob
        rmse = np.sqrt(np.sum(np.square(sea_model-sea))/len(sea))
        return loglikelihood, rmse
    
    else:
        if not np.all(np.isfinite(sea_model)):
            return -np.inf
    
        return loglikelihood
    

#####################################################################
# likelihood function
def lnlike(parameters, sea_std):
    sea_model = jit_modelledVolume(parameters,global_vars)
    if not np.all(np.isfinite(sea_model)):
        return -np.inf, None
    else:
        residuals = np.square((sea-sea_model)/sea_std)
        
        # if gap exists, exclude from likelihood
        if gap!=None:
            # calculate IDs of gap, when to exclude sea level data for tuning
            step = abs(start_year)/time_steps
            gap_start_id = int(abs(start_year-gap[0])/step)
            gap_end_id = int(abs(start_year-gap[1])/step)
            
            # delete gap interval to be not included in tuning
            residuals = np.delete(residuals, range(gap_start_id, gap_end_id+1))
            
        return -0.5*np.sum(residuals), sea_model
        

# flat prior for all parameters (including some bounds)
def lnprior(parameters):
    aEsi, aO, ag, taud, vi, v01, v02, t1, t2, I01, I02, v11, v12 = parameters
    # if (-10.0 < aEsi < 10.0) and (-10.0 < aO < 10.0)  and (-10.0 < ag < 10.0) and (-30.0 < taud0 < 30.0) and (50.0 < v02 < 200.0) and (-50.0 < v1 < 50.0) and (-50.0 < vi < 50.0) and (-30.0 < v01 < 30.0) and (t2 < t1 < -start_year) and (0 < t2 < t1):   
    if (start_year <= t1 < t2 <= 0) and np.all(np.isfinite(parameters)) and np.all(np.array(parameters)>=-1e4) and np.all(np.array(parameters)<=1e4):
        return 0.0

    else:
        return -np.inf
    # if np.all(np.isfinite(parameters)):
    #     return 0.0
    # else:
    #     return -np.inf

# logarithm of the posterior probability
def lnprob(parameters, sea_std):
    lp = lnprior(parameters)
    likelihood, sea_model = lnlike(parameters, sea_std)
    
    if blob:
        if not np.isfinite(lp) or not np.isfinite(likelihood):
            return -np.inf, None
        
        # caluclate RMSE and return as blob
        rmse = np.sqrt(np.sum(np.square(sea_model-sea))/len(sea))
        return lp + likelihood, rmse
        
    else:
        if not np.isfinite(lp) or not np.isfinite(likelihood):
            return -np.inf
        
        return lp + likelihood
    


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
        # default resolution for Berends: 100yr
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

def calc_BIC(parameters, sea, sea_model):
    # Number of data points
    N = len(sea)
    
    # Number of parameters
    n_params = len(parameters)
    
    # calculate log likelihood
    sea_std = np.std(sea)
    LogLikelihood = lnprob(parameters, sea_std)
    
    # BIC
    BIC = -2*LogLikelihood + n_params*np.log(N)
    
    return BIC


# symmetric mean absolute percentage error (SMAPE)    
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))


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
                                        time<=0))
time = -1*time[mask_interval]
esinomega = data_orbital[:,1][mask_interval]#[::-1]
O = data_orbital[:,3][mask_interval]#[::-1]
    

####################################################################
# Interpolate or choose data accordingly to set resolution (Default resolution = Resolution of loaded data = 100 years)
time_sea, sea = np_interpolation(sea, 'sea', resolution, time_sea, sea_data=sea_level_data)  
# sea_std = np_interpolation(sea_std, 'sea', resolution, time_sea, sea_data=sea_level_data)[1]  
# sea_std = np.where(sea_std<1, 1, sea_std)
esinomega = np_interpolation(esinomega, 'esinomega', resolution, time, sea_data=sea_level_data)[1]
time, O = np_interpolation(O, 'O', resolution, time, sea_data=sea_level_data)
             
    
# Prepare Orbital data for model (normalisation, truncation, interpolation)
# Normalization and truncation of parameters input
Esi = normalise(esinomega)
EnO = normalise(O)

#Interpolation to get data at the time step of 500 years (for half-step Runge-Kutta computation)
Esi = interpol(Esi,2)
EnO = interpol(EnO,2)
time_halfsteps = jnp.array(interpol(time,2))
time = jnp.array(time)

global_vars = np.array([Esi, EnO])

# Initial state to 'g'
# state = 0


#####################################################################
#Exploration of the parameters space using a markov chain methodo (MCMC) coupled with a random walk at n walkers using the eemc hammer (Foreman-Mackey, 2013)


# #Define the initial position of each walkers from StartPosition and walkers_jump input values
###############################################################################
# Gaussian initialization
ndim = len(StartPosition)
WalkersIni = np.zeros((nwalkers, ndim))
StdDevParam = np.zeros(len(StartPosition))
for i in range (len(StartPosition)):
    StdDevParam[i] = StartPosition[i]*walkers_jump + 1e-5
        
for j in range(nwalkers):
    for i in range (len(StartPosition)):
        WalkersPos = StartPosition[i] + np.random.normal(0,abs(StdDevParam[i]))
        WalkersIni[j][i] = WalkersPos

# ##############################################################################
# # Uniform initialization
# ndim = len(StartPosition)

# WalkersIni = np.random.uniform(-1_000, 1_000, size=(nwalkers,ndim-2))
# t2s = np.random.uniform(-start_year, 1201, size=(nwalkers,1))
# t1s = np.random.uniform(1200, 0, size=(nwalkers,1))
# WalkersIni = np.hstack((WalkersIni, t2s, t1s))

# ##############################################################################
# # set initial position of last walker to StartPosition        
# WalkersIni[-1] = np.array(StartPosition)


sea_std = np.std(sea)

# MCMC sampling with emcee package for optimization
if tuning=='emcee':
    with Pool() as pool:
        # steps = emcee.EnsembleSampler(nwalkers, ndim, cost_function_negative, args=[sea_std], pool=pool,
        #                                 blobs_dtype=[("rmse", float)],
        #                                 # moves=[(emcee.moves.DESnookerMove(), 0.1), 
        #                                 #       (emcee.moves.DEMove(), 0.9 * 0.9),
        #                                 #       (emcee.moves.DEMove(gamma0=1.0), 0.9 * 0.1)
        #                                 #       ]
        #                                 )
        
        steps = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[sea_std], pool=pool,    
                                        blobs_dtype=[("rmse", float)],
                                        # moves=[(emcee.moves.DEMove(), 0.9),
                                        #         (emcee.moves.DESnookerMove(), 0.1)
                                        #       ]
                                        # moves=[(emcee.moves.DESnookerMove(), 0.1), 
                                        #       (emcee.moves.DEMove(), 0.9 * 0.9),
                                        #       (emcee.moves.DEMove(gamma0=1.0), 0.9 * 0.1)
                                        #        ]
                                        )
        
        steps.run_mcmc(WalkersIni, niterations, progress=True)


# Parallel tempering for optimization
elif tuning=='ptemcee':
    # To avoid a bug with the numpy version we are using, we need to monkey patch `np.float` to `np.float64`
    np.float = np.float64    
    
    def log_prior(params):
        return 0.0
    
    ndim = len(StartPosition)
    
    #Define the initial position of each walkers from StartPosition and walkers_jump input values
    WalkersIni = np.zeros((ntemps, nwalkers, ndim))
    StdDevParam = np.zeros(ndim)
    for i in range(ndim):
        StdDevParam[i] = StartPosition[i]*walkers_jump + 1e-5
            
    for i in range(ntemps):
        for j in range(nwalkers):
            for k in range(ndim):
                WalkersPos = StartPosition[k] + np.random.normal(0,abs(StdDevParam[k]))
                WalkersIni[i][j][k] = WalkersPos
                
    # set initial position of last walker to StartPosition        
    # WalkersIni[-1] = np.array(StartPosition)
    
    with Pool() as pool:
        steps = ptm.Sampler(nwalkers=nwalkers, dim=ndim, ntemps=ntemps, logl=lnprob, loglargs=[sea_std], logp=log_prior, pool=pool)
        steps.run_mcmc(p0=WalkersIni, iterations=niterations, adapt=True, progress=True)

# Nested sampling via dynesty for optimization
elif tuning=='dynesty':
    
    # Define the bounds for each parameter in a dictionary
    bounds = {
        'aEsi': (-1000, 1000),
        'aO': (-1000, 1000),
        'ag': (-1000, 1000),
        'taud': (-1000, 1000),
        'vi': (-500, 500),
        'v01': (-500, 500),
        'v02': (-500, 500),
        't1': (start_year, 0),
        't2': (start_year, 0),
        'I01': (-500, 500),
        'I02': (-500, 500),
        'v11': (-500, 500),
        'v12': (-500, 500)
    }
    
    # static dynesty sampling
    if dynesty_static:
        # Prior transformation: gets parameters uniformly drawn from unit cube;
        # Function that transforms samples from the unit cube to the target prior back
        # UNIFORM PRIORS
        def ptform(theta):
            # Assign theta values
            names = list(bounds.keys())
            transformed = []
            
            for i, name in enumerate(names):
                low, high = bounds[name]
                value = (high - low) * theta[i] + low  # maps [0, 1] to [low, high]
                transformed.append(value)
            
            return transformed
        
        
        # reverse prior transformation: maps parameters to unit cube
        # needed for initalization of live points
        def reverse_ptform(params):
            # List to store transformed values back in the unit cube
            unit_values = []
            names = list(bounds.keys())
            
            for i, name in enumerate(names):
                low, high = bounds[name]
                unit_value = (params[i] - low) / (high - low)  # Normalize to [0, 1]
                unit_values.append(unit_value)
            
            return tuple(unit_values)
    
    # dynamic dynesty sampling
    else:
        # TRUNCATED NORMAL PRIORS
        def ptform(theta):
            # # Assign theta values
            # names = list(bounds.keys())
            # transformed = []
            
            # for i, name in enumerate(names):
            #     mean, std = StartPosition[i], StartPosition[i]*0.1+1e-5  # mean and standard deviation
            #     low, high = bounds[name]
            #     low_n, high_n = (low-mean)/std, (high-mean)/std  # standardize
                
            #     value = scipy.stats.truncnorm.ppf(theta[i], low_n, high_n, loc=mean, scale=std) 
            #     transformed.append(value)
            
            # Assign theta values
            transformed = []
            
            for i in range(len(theta)):
                    value = StartPosition[i] + (StartPosition[i]*0.1+1e-5)*scipy.special.ndtri(theta[i])
                    transformed.append(value)
                
            return transformed
    
    # Normalization for log likelihood
    sea_std = np.std(sea)
    # lnorm = -0.5 * (np.log(2 * np.pi) * len(sea) + np.log(sea_std**2)) 
    lnorm = np.log(2 * np.pi * sea_std**2)
    
    # Log Likelihood function
    def loglike(theta):
        sea_model = jit_modelledVolume(theta, global_vars)
        if not np.all(np.isfinite(sea_model)):
            ll = -np.inf
        else:
            residuals = np.square((sea-sea_model)/sea_std)
            
            # if gap exists, exclude from likelihood
            if gap!=None:
                # calculate IDs of gap, when to exclude sea level data for tuning
                step = abs(start_year)/time_steps
                gap_start_id = int(abs(start_year-gap[0])/step)
                gap_end_id = int(abs(start_year-gap[1])/step)
                
                # delete gap interval to be not included in tuning
                residuals = np.delete(residuals, range(gap_start_id, gap_end_id+1))
                
            ll = -0.5*np.sum(residuals+lnorm)
        
        return ll
    
    # set initial points (only for static nested sampling possible)
    if dynesty_static:
        # Initilize live points: Gaussian initialization (in normal space)
        ndim = len(StartPosition)
        live_v = np.zeros((nlive, ndim))
        StdDevParam = np.zeros(len(StartPosition))
        for i in range (len(StartPosition)):
            StdDevParam[i] = StartPosition[i]*walkers_jump + 1e-5
                
        for j in range(nlive):
            for i,key in enumerate(bounds.keys()):
                LivePos = -np.inf
                while LivePos<bounds[key][0] or LivePos>bounds[key][1]:
                    LivePos = StartPosition[i] + np.random.normal(0,abs(StdDevParam[i]))
                # add initial point if within bounds of parameters
                live_v[j][i] = LivePos
                
        # transform live points into unit cube
        live_u = np.zeros((nlive,ndim))
        for i in range(nlive):
            live_u[i] = reverse_ptform(live_v[i,:])
            
        # associated log likelihoods of inital live points
        live_logl = np.zeros(nlive)
        for i in range(nlive):
            live_logl[i] = loglike(live_v[i,:])
        
    
    ndim = len(StartPosition)
    
    # static sampling
    if dynesty_static:
        with dynesty.pool.Pool(20, loglike, ptform) as pool:
            sampler = dynesty.NestedSampler(loglike, ptform, ndim, pool=pool, nlive=nlive
                                            , sample='slice'  #slice
                                            , bound='multi'
                                            # , walks=nwalkers
                                            # , facc=0.5
                                            , live_points=[live_u, live_v, live_logl]
                                            )
            sampler.run_nested(maxiter=niterations, dlogz=-1)  
            # sampler.run_nested(maxiter=niterations, use_stop=False)
            steps = sampler.results
    
    # dynamic sampling
    else:
        with dynesty.pool.Pool(20, loglike, ptform) as pool:
            sampler = dynesty.DynamicNestedSampler(loglike, ptform, ndim, pool=pool, nlive=nlive, sample='slice', bound='multi')
            sampler.run_nested(maxiter=niterations, use_stop=False)  
            # sampler.run_nested(maxiter=niterations, use_stop=False)
            steps = sampler.results
        
        
    N = len(steps.samples)
    rmses = []
    for i in tqdm(range(N)):
        icevolume = jit_modelledVolume(steps.samples[i,:], global_vars)
        rmse = np.sqrt(np.sum(np.square(icevolume-sea))/len(sea))
        rmses.append(rmse)
    print('\nMinimal RMSE: ',np.min(rmses))
    
    # print best params
    flat_logprob = steps.logl
    best_index = np.argmax(flat_logprob)
    best_likelihood = flat_logprob[best_index]
    best_params = steps.samples[best_index].tolist()
    print(f'Best params: {best_params}')
    
    # print summary
    steps.summary()
    
    # Plot results
    dyplot.cornerplot(steps, labels=parameter_names, show_titles=True, color='blue');
    dyplot.traceplot(steps, labels=parameter_names);
    
    # Plot arviz traceplot
    # Assuming 'steps.samples' contains the samples from dynesty, with shape (draws, parameters)
    samples = steps.samples

    # Convert samples to a dictionary for arviz
    data_dict = {name: samples[:, i] for i, name in enumerate(parameter_names)}

    # Wrap the dictionary in an InferenceData object
    idata = az.from_dict(posterior=data_dict)

    # Plot the trace
    az.plot_trace(idata)
    
# PyMC for optimization
else:
    #####################################################################
    # Set up the OP
    # The CustomOp needs `make_node` and `perform`.
    class CustomOp(Op):
        def make_node(self, parameters, global_vars, sea):
            # Create a PyTensor node specifying the number and type of inputs and outputs

            # We convert the input into a PyTensor tensor variable
            parameters = pt.as_tensor_variable(parameters)
            global_vars = pt.as_tensor_variable(global_vars)
            sea = pt.as_tensor_variable(sea)
            
            # inputs = [parameters, sea]
            inputs = [parameters, global_vars, sea]
            
            # Output has the same type and shape as `x`
            # outputs = [inputs[0][0].type()]
            outputs = [pt.as_tensor(sea).type()]
            
            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            # Evaluate the Op result for a specific numerical input

            # The inputs are always wrapped in a list
            parameters, global_vars, sea = inputs
            
            # result = lnprob(parameters, sea)
            result = jit_modelledVolume(parameters, global_vars)
            if gap!=None:
                # calculate IDs of gap, when to exclude sea level data for tuning
                step = abs(start_year)/time_steps
                gap_start_id = int(abs(start_year-gap[0])/step)
                gap_end_id = int(abs(start_year-gap[1])/step)
                
                # delete gap interval to be not included in tuning
                result = np.delete(result, range(gap_start_id, gap_end_id+1))
                
            # The results should be assigned inplace to the nested list
            # of outputs provided by PyTensor. If you have multiple
            # outputs and results, you should assign each at outputs[i][0]
            outputs[0][0] = np.asarray(result, dtype="float64")   #, dtype="float64"

    # Instantiate the Ops
    custom_op = CustomOp()


    ###############################################################################
    # Set up PyMC model
    initvals = {'params': StartPosition}

    bounds = {'lower': [-1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000,   start_year,      -1_000,    -1_000, -1_000],
              'upper': [ 1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,     -1_001,           0,       1_000,  1_000]
              }

    coords = {'time': time, 'parameters': parameter_names}

    # use PyMC to sampler from log-likelihood
    StartPosition_dict = dict(zip(parameter_names, StartPosition))

    if gap!=None:
        # calculate IDs of gap, when to exclude sea level data for tuning
        step = abs(start_year)/time_steps
        gap_start_id = int(abs(start_year-gap[0])/step)
        gap_end_id = int(abs(start_year-gap[1])/step)
        
        # delete gap interval to be not included in tuning
        sea_data = np.delete(sea, range(gap_start_id, gap_end_id+1))
    else:
        sea_data = sea
        
    with pm.Model(coords=coords) as model:
        # data 
        sea_data_pymc = pm.ConstantData('sea_data', sea_data)
        
        # priors
        # all params together
        params = pm.TruncatedNormal('params', mu=StartPosition, sigma=walkers_jump*np.abs(StartPosition)+1e-5, dims='parameters', 
                                    lower=bounds['lower'], upper=bounds['upper'], initval=StartPosition) #, initval=StartPosition
        
        sigma = pm.HalfNormal("sigma", sigma=15) 
        
        # model
        # mu = pm.Deterministic("mu", custom_op(params, sea_data_pymc))
        sea_model = pm.Deterministic("sea_model", custom_op(params, global_vars, sea_data))
        rmse = pm.Deterministic('rmse', pm.math.sqrt(pm.math.sum((sea_model-sea_data_pymc)**2)/len(sea_data)))

        # posterior
        y = pm.Normal('y', mu=sea_model, sigma=sigma, observed=sea_data_pymc)


    ###############################################################################
    # sample from the PyMC model

    with model:
        # steps = pm.step_methods.DEMetropolis()
        steps = pm.step_methods.Metropolis()
        idata = pm.sample(niterations, tune=niterations,  chains=4, cores=4, discard_tuned_samples=False, step=steps, initvals=initvals)
        

    ##############################################################################
    # create a trace plot
    # plot the traces
    az.plot_trace(idata, 
                  var_names=['params','rmse'],
                  compact=False,);

#####################################################################
#Extraction of the best parameters list to copy and paste in the GRAD_simulation_plot.py program

#Recovering of parameters from which we obtain the minimum residuals
# flat_chain = steps.flatchain
# best_vars = flat_chain[np.argmax(steps.flatlnprobability)]


if tuning=='ptemcee':
    chain = steps.chain
    logprob = steps.loglikelihood
    best_index = np.unravel_index(np.argmax(logprob), logprob.shape)
    
    best_params = chain[best_index].tolist()
    best_likelihood = np.max(logprob)
    
elif tuning=='emcee':
    flat_logprob = steps.get_log_prob(flat=True)
    flat_chain = steps.get_chain(flat=True)
    best_index = np.argmax(flat_logprob)
    best_likelihood = np.max(flat_logprob)
    best_params = flat_chain[best_index].tolist()
    
    if blob:
        flat_blobs = steps.get_blobs(flat=True)
        flat_rmses = flat_blobs['rmse']
        blob_min_rmse = np.nanmin(flat_rmses)
        
elif tuning=='dynesty':
    flat_logprob = steps.logl
    best_index = np.argmax(flat_logprob)
    best_likelihood = flat_logprob[best_index]
    best_params = steps.samples[best_index].tolist()
        
else:
    ##############################################################################
    # identify smallest RMSE and best params
    rmses = idata.posterior.rmse.values
    min_rmse = np.min(rmses)

    # Find the flattened index of the minimum value
    flat_argmin = np.argmin(rmses)

    # Convert the flattened index back to the multi-dimensional index (chain, draw)
    chain_idx, draw_idx = np.unravel_index(flat_argmin, rmses.shape)

    # Get best params
    best_params = idata.posterior.params[chain_idx, draw_idx, :].values.tolist()

    # print(f"Argmin is at chain: {chain_idx}, draw: {draw_idx}")
    # print(f"Minimal RMSE: {min_rmse}")
    # print(f'Best params: {best_params}')

    
    
# calculate RMSE, MAE, R2, SMAPE and BIC 
icevolume = jit_modelledVolume(best_params, global_vars)

rmse = root_mean_squared_error(y_true=sea, y_pred=icevolume)
if gap!=None:
    residuals_gap = (sea-icevolume)**2
    # calculate IDs of gap, where to exclude for RMSE
    step = abs(start_year)/time_steps
    gap_start_id = int(abs(start_year-gap[0])/step)
    gap_end_id = int(abs(start_year-gap[1])/step)
    
    # delete gap interval to be not included in RMSE
    residuals_gap = np.delete(residuals_gap, range(gap_start_id, gap_end_id+1))
    
    gap_rmse = np.sqrt(np.sum(residuals_gap)/len(residuals_gap))
    
mae = mean_absolute_error(y_true=sea, y_pred=icevolume)
R2 = r2_score(y_true=sea, y_pred=icevolume)
SMAPE = smape(y_true=sea, y_pred=icevolume)
BIC = calc_BIC(best_params, sea, icevolume)
     
print(f"RMSE = {rmse}")
if gap!=None:
    print(f"RMSE (Gap) = {gap_rmse}")
if tuning!='pymc':
    print(f"Best likelihood = {best_likelihood}")  
print(f"MAE = {mae}")
print(f"R² = {R2}")
print(f"BIC = {BIC}")
print(f"Best fit parameters are : {best_params}")    
if blob:
    print(f'Minimum RMSE(blob)= {blob_min_rmse}')

tac = time_module.perf_counter() 
dtime = (tac-tic) 
print(f"Execution time: {dtime:.4e} seconds")
print(f"Execution time: {dtime/60:.1f} minutes")


###############################################################################
### Some extra plots
###############################################################################
import seaborn as sns
import matplotlib.pyplot as plt

###############################################################################
print('\nPlotting the histogram of the residuals of the best model...')
residuals = icevolume-sea

sns.histplot(residuals, bins=50, kde=True, color='b', edgecolor='black')
plt.title('For best model\nResiduals=icevolume-sea')
plt.show()

# ###############################################################################
# print('\nPlotting the RMSE histograms of whole chain...')
# if paralleltuning=='ptemcee':
#     chain = steps.flatchain.reshape(-1, steps.flatchain.shape[-1])
# else:
#     chain = steps.get_chain(flat=True, discard=False)
# RMSEs = []
# for i in tqdm.tqdm(range(chain.shape[0])):
#     icevol = modelledVolume(start_year,0,chain[i,11],state,global_vars,chain[i],time_steps)
#     rmse = np.sqrt(np.sum(np.square(icevol-sea))/len(sea)) 
#     RMSEs.append(rmse)

# plt.hist(RMSEs, bins=100, range=(np.min(RMSEs), 25))
# plt.title('Histograms of RMSEs of whole chain')
# plt.show()


###############################################################################
if gap==None:
    print('\nPlotting 100 residuals of last 10% of chain...')
    fig, ax = plt.subplots(figsize=(20,5))
    fig.tight_layout(pad=6.0)
    
    if tuning=='ptemcee':
        # use only first temperature (beta=1), since there chain is sampling posterior
        flat_chain = steps.flatchain[0,:][int(0.9*niterations*nwalkers):,:]
    elif tuning=='emcee':
        flat_chain = steps.get_chain(flat=True, discard=int(0.9*niterations), thin=nwalkers)
    elif tuning=='dynesty':
        flat_chain = steps.samples[int(0.9*steps.niter):, :]
    else:
        # get last 10% from best chain
        flat_chain = idata.posterior.sea_model[chain_idx, int(0.9*niterations):].values.reshape((-1, len(sea)))
    
    if tuning!='pymc':
        for params in flat_chain[np.random.randint(len(flat_chain), size=100)]:
            icevol = jit_modelledVolume(params, global_vars)
            ax.plot(time, icevol, color='grey', alpha=0.1)
    else:
        for icevol in flat_chain[np.random.randint(len(flat_chain), size=100)]:
            ax.plot(time, icevol, color='grey', alpha=0.1)
            
    # for i in tqdm.tqdm(range(chain.shape[0])):
    #     icevol = modelledVolume(start_year,0,chain[i,11],state,global_vars,chain[i],time_steps)
    #     ax.plot(time, icevol, color='grey', alpha=0.05)
    
    ax.plot(time_sea, sea, color='blue', label="Berends sea level data")
    ax.plot(time, icevolume, color='black', label="Best model")
    
    plt.xlim(-start_year,0)
    ax.set_ylim(np.min([icevolume,sea])-0.1*np.absolute(np.max(icevolume)),np.max([icevolume,sea])+0.1*np.absolute(np.max(icevolume)))
    ax.invert_yaxis()
    ax.set_xlabel("Age (ka)",weight='bold')
    ax.set_ylabel("Ice volume (m sl)",weight='bold')
    
    # plt.savefig('../Data/RAMP-2.png', dpi=500, bbox_inches='tight')
    plt.show()

# ###############################################################################
# print('\nPlotting the median-spread of last 10% of chain...')

# # calculates median + spread of chain (based on nsamples drawn from last 50% of chain)
# def sample_walkers(nsamples, steps):
#     flat_chain = steps.get_chain(flat=True, discard=int(niterations/2))
#     icevolumes = []
    
#     random_samples = np.floor(np.random.uniform(0,len(flat_chain),size=nsamples)).astype(int)
#     params_list = flat_chain[random_samples]
    
#     for params in params_list:
#         icevol = modelledVolume(params)
#         icevolumes.append(icevol)
        
#     std_icevol = np.std(icevolumes, axis=0)
#     med_icevol = np.median(icevolumes, axis=0)
    
#     return med_icevol, std_icevol

# med_icevol, std_icevol = sample_walkers(nsamples=1_000, steps=steps)


# fig, ax = plt.subplots(figsize=(20,5))
# fig.tight_layout(pad=6.0)

# plt.fill_between(time, med_icevol-std_icevol, med_icevol+std_icevol, color='grey', alpha=0.5, 
#                  label=r'$1\sigma$ Posterior Spread')
# ax.plot(time, sea, color='blue', label="Berends sea level data")
# ax.plot(time, icevolume, color='black', label="Best model")

# plt.xlim(-start_year,0)
# ax.set_ylim(np.min([icevolume,sea])-10,np.max([icevolume,sea])+10)
# ax.invert_yaxis()
# ax.set_xlabel("Age (ka)",weight='bold')
# ax.set_ylabel("Ice volume (m sl)",weight='bold')

# plt.show()
                        
            












