#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:29:13 2024

@author: pollakf

Note:  - Based on RAMP7_jax.py, but changed for GRAD model 
    
       - Switch included to switch between normal and intersection-based model
         (if switch false: normal GRAD model /wo intersects)
       - Calculations to spot intersection points at boundary of v(t) 
         and threshold functions
       - GAPs to verify tuning
       - PyMC + dynesty tuning added (switch to decide between emcee/ptemcee/PyMC)
       - CURRENT STANDARD GRAD MODEL!
      
Note2: JAX uses by default float32. float64 is much slower and has to be 
       explicitly enabled
       
"""


import numpy as np 
import emcee
from multiprocessing import Pool
import time as time_module 
from MyModules import ptemcee_modified as ptm  # ptemcee modified to include progress bar
from numba import njit
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import pandas as pd
from tqdm import tqdm
import dynesty
from dynesty import plotting as dyplot
import os
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


tic = time_module.perf_counter() 

###############################################################################
# BERENDS SEA LEVEL DATA
###############################################################################

###############################################################################
# Berends [-2 Myr - 0], no gap, no intersections
###############################################################################

# Parameters from Pollak et al. (2025)
# StartPosition = (-0.2827904281673865, -0.23791176812137232, 0.5582094270909611, 0.9166431597555614, 1.122481747544036, 3.785487641389463, 18.108000606742888, -1.6932954204268298, 15.749275998849093, 121.95347895900386, 8.696160394087554, 15.991220441376605, 0.03971613203152333, -0.002368159408302358)
# RMSE = 11.709785697631608

# Improved parameters
# StartPosition = (-0.276255998318236, -0.32378708383973276, 0.5901991614742599, 0.9194973941634998, 1.0211072609370149, 3.9409329051925397, 15.362913050575912, -1.5489857755761705, 13.354251737397703, 114.65542239706093, 7.460847619780793, 15.727425853504542, 0.0364932933383173, -0.0023776043602339314)
# RMSE = 11.67107679732531

# StartPosition = (0.5, 0.5, 0.5, 0.5, 0.5, 10, 1, 1, 1, 100, 1, 1, 0.01, -0.01)

###############################################################################
# Berends [-2.6 Myr - 0], no gap, no intersections
###############################################################################

# Parameters from Pollak et al. (2025)
StartPosition = (-0.2796942938131826, -0.41055243998242824, 0.7678233937082588, 0.9034653801264746, 0.1735604154421203, 5.230136792609927, 21.893854799971354, 7.709522263449515, 12.455945514056516, 126.35731589261985, -10.470994711607666, 9.275583978063581, 0.040873841850912196, -0.005617981199517685)
# RMSE = 11.857648096170037

# Improved parameters
# StartPosition = (-0.22506144282443863, -0.41030076388383563, 0.7746134819186494, 0.8862232286060134, 0.15000967736216353, 5.278874681030402, 21.58678541430686, 7.582127649751442, 11.994511125223084, 125.61999213424883, -10.230567755039488, 10.102938865998908, 0.04055342061799294, -0.005988987069645901)
# RMSE = 11.833832612935126

###############################################################################
# Berends [-3.6 Myr - 0], no gap, no intersections
###############################################################################

# StartPosition = (-0.24486896956592766, -0.2889143057790853, 0.44000443418484686, 0.9625924013694424, -0.16907325531696404, 1.1167903183156123, 16.312992788472176, 1.0571889988340368, 8.946710967989533, 127.0841874412489, -0.1877471670950272, -14.211442158052444, 0.05140227479111467, -0.010105090373309134)
# RMSE = 11.168642549069645


###############################################################################
# ROHLING SEA LEVEL DATA
###############################################################################

###############################################################################
# Rohling [-2.6 Myr - 0], no gap, no intersections
###############################################################################

# StartPosition = (0.6131249146679715, -6.459565879851357e-05, 1.108367996066035, 0.7780442555153607, 1.9527786059589782, 6.801644117993495, 7.425336607752293, 15.288550600276254, 6.343495087038375, 115.35414684977087, -4.308405491626213, 19.375533951158374, 0.03599550380697811, -0.0006793476890781925)
# RMSE = 12.254415018699635


# Parameters = [aEsi, aEco, aO, ag, ad, tau_d, kEsi, kEco, kO, v0, v1, vi, C_v0, C_taud]
#                0      1    2   3   4    5     6     7    8   9   10  11   12    13

parameter_names = ['aEsi', 'aEco', 'aO', 'ag', 'ad', 'tau_d', 'kEsi', 'kEco', 'kO', 'v0', 'v1', 'vi', 'C_v0', 'C_taud']


   
#Number of walkers (verifying ; nwalkers > 2 * number of parameters)
nwalkers = 100

#Number of iterations 
niterations = 1_000   #100_000   (for 1000yr res: 500_000 + walker_jumps=0.3)

#Define the first position of each walkers relatively to StartPosition. When walkers_jump is high, walkers are far from StartPosition. 
walkers_jump = 0.01

# Number of live points (only needed for dynesty sampler)
nlive = 1_024

# Number of temperatures (only for parallel tempering)
ntemps = 50

# Set resolution of model in yr
resolution = 1000

# Start year in kyr 
start_year = -int(2_000)

# Do not change (ends simulation at present time 0) 
future_time = int(0)

# Gap included for tuning: Model is not tuned during this time interval. gap=(start_gap[kyr BP], end_gap[kyr BP])
# gap = (-int(1_200), -int(700)) 
# gap = (-int(130), -int(0)) 
# gap = (-int(2600), -int(700))
gap = None

time_steps = int(-start_year*1e3/resolution)  # number of timesteps in model

# Select tuning procedure
tuning = 'emcee'  # Options: 'emcee', 'ptemcee', 'pymc', 'dynesty'


# switch for intersections
intersections = False


# Set sea-level data: either Berends or Rohling (LR04 based + tuned age) 
sea_level_data = 'Berends'

blob = False
if tuning!='emcee':
    blob = False

print(f"Model resolution: {resolution} years")
print(f"Number of timesteps: {time_steps}")   
print(f"Number of iterations: {niterations}") 
print(f"Simulated time interval: {start_year} kyr BP - present")
if gap!=None:
    print(f"Gap for tuning: {gap[0]} kyr - {gap[1]} kyr")
print(f"Numerical scheme with intersections: {intersections}")   
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
    Eco = global_vars[1]
    EnO = global_vars[2]
    
    # current time t
    t = -1 * start_year - ((i * sim_time / time_steps) / 2)
    
    # calculate tau_d(t) and take care for division by 0
    taud_t = params[5] - params[13] * t
    taud_t = jnp.where(taud_t == 0, 1e-8, taud_t) 

    # Calculate change in ice volume dv/dt
    dvdt = jnp.where(state == 0,
                     -params[0] * Esi[i] - params[1] * Eco[i] - params[2] * EnO[i] + params[3],
                     -params[0] * Esi[i] - params[1] * Eco[i] - params[2] * EnO[i] + params[4] - vt / taud_t)
    
    return dvdt

# calculate additional intersection points
if intersections:
    #####################################################################
    # gradient implementation to be compatible with JAX (jnp.gradient doesn't work)
    # grad has same shape as input array. Same implementation as for np.gradient:
    # central first-order gradient  
    # !!! very important to keep the resolution in mind, also if there is a difference for vt and v0_bounds
    def np_gradient(x):
        # Central difference for internal points, forward difference for the first point, backward for the last
        grad = jnp.zeros_like(x)
        grad = grad.at[1:-1].set((x[2:] - x[:-2]) / (2*resolution*1e-3))
        grad = grad.at[0].set(x[1] - x[0])
        grad = grad.at[-1].set(x[-1] - x[-2])
        return grad
    
    #######################################################
    # calculate lower and upper bound for v, where deglaciation/glaciation starts
    def calc_bounds(params, global_vars, time_steps):
        Esi = global_vars[0]
        Eco = global_vars[1]
        EnO = global_vars[2]
        
        v0s = jnp.zeros(time_steps+1)
        v0_bounds = jnp.zeros(time_steps+1)
        v1_bounds = jnp.zeros(time_steps+1)
        # v0_bound = v0(t)-insolation (eq.5 from Legrain et.al.)
        
        # complete simulation time
        sim_time = jnp.abs(start_year)
        
        # body function for JAX loop
        def body_fn(i, val):
            v0s, v0_bounds, v1_bounds = val
            
            # current time t
            t = -1 * start_year - (i * sim_time / time_steps)
            
            # calculate v_0(t)
            v0_t = params[9] - params[12] * t
            
            # Update bounds
            v0s = v0s.at[i].set(v0_t)
            v0_bounds = v0_bounds.at[i].set(v0_t - params[6] * Esi[2*i] - params[7] * Eco[2*i] - params[8] * EnO[2*i])
            v1_bounds = v1_bounds.at[i].set(params[6] * Esi[2*i] + params[7] * Eco[2*i] + params[8] * EnO[2*i])
            
            return v0s, v0_bounds, v1_bounds
        

        # Use JAX loop to handle iteration
        v0s, v0_bounds, v1_bounds = jax.lax.fori_loop(lower=0, upper=time_steps+1, body_fun=body_fn, init_val=(v0s, v0_bounds, v1_bounds))
        
        return v0s, v0_bounds, v1_bounds
    
    #####################################################################
    # Function to calculate intersection points between v(t) and deglac. threshold (based on v0)
    def calc_intersect_v0(i, v, v0_bounds, v0_bounds_grad, time):
        v_grad = np_gradient(v)
        
        # calculate intersection
        intersect = -(v[i-1]-v0_bounds[i-1])/(v_grad[i-1]-v0_bounds_grad[i-1])
        
        return intersect, time[i-1]-intersect


    #####################################################################
    # Function to calculate intersection points between insolation and v1
    def calc_intersect_v1(i, v1, v1_bounds, v1_bounds_grad, time):
        # calculate intersection
        intersect = (v1-v1_bounds[i-1]) / v1_bounds_grad[i-1]
        
        return intersect, time[i-1]-intersect


    #####################################################################
    # Function to calculate v(i+t') at intersection point and recalculate v(i) after intersection point
    def intersection(i, v, v1, state, params, v0_bounds, v1_bounds, v0_bounds_grad, v1_bounds_grad, time, sim_time, v0_intersection, time_halfsteps, global_vars):
        # Conditional intersection calculation based on v0_intersection
        intersect, time_intersect = jax.lax.cond(
            v0_intersection,
            lambda: calc_intersect_v0(i, v, v0_bounds, v0_bounds_grad, time),   # calculate intersection point between v(t) and degl. threshold (based on v0)
            lambda: calc_intersect_v1(i, v1, v1_bounds, v1_bounds_grad, time)   # calculate intersection point between insolation and v1
        )
        
        # calculate v at t+t' (intersection point)
        j = i-1
        k1 = Phi(2*j, v[j], params, state, global_vars, sim_time, time_steps)
        k2 = Phi_intersect(vt=v[j]+k1*intersect/2, time_intersect=time[j]-intersect/2, params=params, state=state, global_vars=global_vars, time_halfsteps=time_halfsteps)
        k3 = Phi_intersect(vt=v[j]+k2*intersect/2, time_intersect=time[j]-intersect/2, params=params, state=state, global_vars=global_vars, time_halfsteps=time_halfsteps)
        k4 = Phi_intersect(vt=v[j]+k3*intersect, time_intersect=time_intersect, params=params, state=state, global_vars=global_vars, time_halfsteps=time_halfsteps)
        v_intersect = v[j] + intersect/6.*(k1+2*k2+2*k3+k4)
        
        # change state
        state = jnp.where(state == 0, 1, 0)
        
        # recalculate v at full position i (t+step). In "d" state 
        step_after_intersect = jnp.abs(time[i]-time_intersect)  # time step between t_intersect and t[i] (after intersection)
        k1 = Phi_intersect(vt=v_intersect, time_intersect=time_intersect, params=params, state=state, global_vars=global_vars, time_halfsteps=time_halfsteps)
        k2 = Phi_intersect(vt=v_intersect+k1*step_after_intersect/2, time_intersect=time_intersect-step_after_intersect/2, params=params, state=state, global_vars=global_vars, time_halfsteps=time_halfsteps)
        k3 = Phi_intersect(vt=v_intersect+k2*step_after_intersect/2, time_intersect=time_intersect-step_after_intersect/2, params=params, state=state, global_vars=global_vars, time_halfsteps=time_halfsteps)
        k4 = Phi(2*i, v_intersect+step_after_intersect*k3, params, state, global_vars, sim_time, time_steps)
        v = v.at[i].set(v_intersect + step_after_intersect / 6. * (k1 + 2*k2 + 2*k3 + k4))
        
        # now both thresholds crossed
        g_threshold_gd = jnp.where(v0_intersection, 1, 0)
        g_threshold_dg = jnp.where(v0_intersection, 1, 0)
        d_threshold_gd = jnp.where(v0_intersection, 0, 1)
        d_threshold_dg = jnp.where(v0_intersection, 0, 1)
            
        return v, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
    #######################################################
    #Compute the derivative at intersection points
    def Phi_intersect(vt, params, state, global_vars, time_intersect, time_halfsteps):
        Esi = global_vars[0]
        Eco = global_vars[1]
        EnO = global_vars[2]
        
        # calculate tau_d(t) and take care for division by 0
        taud_t = params[5] - params[13] * time_intersect
        taud_t = jnp.where(taud_t == 0, 1e-8, taud_t) 
    
        # calculate Esi, Eco and EnO at intersection point
        Esi_intersect = jnp.interp(time_intersect, time_halfsteps[::-1], Esi[::-1])
        Eco_intersect = jnp.interp(time_intersect, time_halfsteps[::-1], Eco[::-1])
        EnO_intersect = jnp.interp(time_intersect, time_halfsteps[::-1], EnO[::-1])
        
        # Calculate change in ice volume dv/dt
        dvdt = jnp.where(state == 0,
                         -params[0] * Esi_intersect - params[1] * Eco_intersect - params[2] * EnO_intersect + params[3],
                         -params[0] * Esi_intersect - params[1] * Eco_intersect - params[2] * EnO_intersect + params[4] - vt/taud_t)
        
        return dvdt
    
    # Compute the modelled volume for a set of input parameters using the Runge–Kutta 4th order method
    def modelledVolume(params, global_vars):
        state = 0  # 'g' state represented as 0
        g_threshold_gd = 0
        g_threshold_dg = 0
        d_threshold_gd = 0
        d_threshold_dg = 0
        
        params = jnp.array(params)
        global_vars = jnp.array(global_vars)
        vi = params[11]
        Esi = global_vars[0]
        Eco = global_vars[1]
        EnO = global_vars[2]
        
        # calculate thresholds and gradients
        v0s, v0_bounds, v1_bounds = calc_bounds(params, global_vars, time_steps)
        v0_bounds_grad = np_gradient(v0_bounds)
        v1_bounds_grad = np_gradient(v1_bounds)
        
        # total simulation time
        sim_time = jnp.abs(start_year) 
        
        
        vt = jnp.zeros(time_steps + 1)
        vt = vt.at[0].set(vi)
        step = -start_year/float(time_steps)
        
        def body(i, val):
            vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = val
            
            # current time t (full timesteps only)
            t = -1 * start_year - (i * sim_time / time_steps)
            
            # thresholds for state changes (use Esi, Eco, EnO at full time steps only)
            test_threshold_gd = params[6] * Esi[2*i] + params[7] * Eco[2*i] + params[8] * EnO[2*i] + vt[i]
            test_threshold_dg = params[6] * Esi[2*i] + params[7] * Eco[2*i] + params[8] * EnO[2*i]
    
            # calculate v_0(t)
            v0_t = params[9] - params[12] * t
            
            
            # check for state change and interesction in glacial state 
            def if_glacial(vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg):
                def state_change(vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg):
                    # only calculate intersections, if v(t) crossed gd_threshold, but insolation> v1 was already fulfilled at last timestep
                    def intersection_case_1():
                        new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=vt, v1=params[10], params=params, state=state, v0_bounds=v0_bounds, v1_bounds=v1_bounds, v0_bounds_grad=v0_bounds_grad, v1_bounds_grad=v1_bounds_grad, time=time, sim_time=sim_time, v0_intersection=1, time_halfsteps=time_halfsteps, global_vars=global_vars)    
                        return new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
                    
                    # only calculate intersections, if insolation crossed v1 threshold, but v(t)<gd_threshold was already fulfilled at last timestep
                    def intersection_case_2():
                        new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=vt, v1=params[10], params=params, state=state, v0_bounds=v0_bounds, v1_bounds=v1_bounds, v0_bounds_grad=v0_bounds_grad, v1_bounds_grad=v1_bounds_grad, time=time, sim_time=sim_time, v0_intersection=0, time_halfsteps=time_halfsteps, global_vars=global_vars)
                        return new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
                        
                    # if both thresholds were crossed at same timestep
                    def intersection_case_3():
                        # calculate both intersection points
                        intersect_v0, time_v0 = calc_intersect_v0(i, vt, v0_bounds, v0_bounds_grad, time)
                        intersect_v1, time_v1 = calc_intersect_v1(i, params[10], v1_bounds, v1_bounds_grad, time)
                        
                        # choose the later (=smaller) intersection point
                        new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = jax.lax.cond(
                            time_v0 <= time_v1, 
                            lambda: intersection(i=i, v=vt, v1=params[10], params=params, state=state, v0_bounds=v0_bounds, v1_bounds=v1_bounds, v0_bounds_grad=v0_bounds_grad, v1_bounds_grad=v1_bounds_grad, time=time, sim_time=sim_time, v0_intersection=1, time_halfsteps=time_halfsteps, global_vars=global_vars), 
                            lambda: intersection(i=i, v=vt, v1=params[10], params=params, state=state, v0_bounds=v0_bounds, v1_bounds=v1_bounds, v0_bounds_grad=v0_bounds_grad, v1_bounds_grad=v1_bounds_grad, time=time, sim_time=sim_time, v0_intersection=0, time_halfsteps=time_halfsteps, global_vars=global_vars)
                            )
    
                        # now both thresholds crossed
                        g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = (1, 1, 0, 0)
                        
                        return new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
                    
                    # Check which transition occurs and if needed calculate intersection point
                    vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = jax.lax.cond(
                        g_threshold_dg & (jnp.logical_not(g_threshold_gd)),
                        intersection_case_1,  # Case 1
                        lambda : jax.lax.cond(
                            g_threshold_gd & (jnp.logical_not(g_threshold_dg)),
                            intersection_case_2,  # Case 2
                            intersection_case_3   # Case 3: both thresholds crossed
                        )
                    )
                    
                    return vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
                    
                def no_state_change():
                    return vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
                    
                vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = jax.lax.cond(
                                (test_threshold_gd > v0_t) & (test_threshold_dg > params[10]),
                                lambda: state_change(vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg),
                                no_state_change
                            )
                
                # if only one condition holds, store which one (at step i)
                # needed for step i+1, when both thresholds are fulfilled, but need 
                # to know which of these conditions was previously false at i
                g_threshold_gd = jnp.where(test_threshold_gd>v0_t, 1, 0)
                g_threshold_dg = jnp.where(test_threshold_dg>params[10], 1, 0)
                    
                return vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
    
            
            # check for state change and interesction in deglacial state 
            def if_deglacial(vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg):
                def state_change(vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg):
                    # only calculate intersections, if insolation gets smaller v1, but v(t) was already smaller degl. threshold at last timestep
                    def intersection_case_1():
                        new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=vt, v1=params[10], params=params, state=state, v0_bounds=v0_bounds, v1_bounds=v1_bounds, v0_bounds_grad=v0_bounds_grad, v1_bounds_grad=v1_bounds_grad, time=time, sim_time=sim_time, v0_intersection=0, time_halfsteps=time_halfsteps, global_vars=global_vars)
                        return new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
                    
                    # only calculate intersections, if v(t) gets smaller than degl. threshold, but insolation was already smaller than v1 at last timestep
                    def intersection_case_2():
                        new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=vt, v1=params[10], params=params, state=state, v0_bounds=v0_bounds, v1_bounds=v1_bounds, v0_bounds_grad=v0_bounds_grad, v1_bounds_grad=v1_bounds_grad, time=time, sim_time=sim_time, v0_intersection=1, time_halfsteps=time_halfsteps, global_vars=global_vars)        
                        return new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
                        
                    # if both thresholds were crossed at same timestep
                    def intersection_case_3():
                        # calculate both intersection points
                        intersect_v0, time_v0 = calc_intersect_v0(i, vt, v0_bounds, v0_bounds_grad, time)
                        intersect_v1, time_v1 = calc_intersect_v1(i, params[10], v1_bounds, v1_bounds_grad, time)
                        
                        # choose the later (=smaller) intersection point
                        new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = jax.lax.cond(
                            time_v0 <= time_v1, 
                            lambda: intersection(i=i, v=vt, v1=params[10], params=params, state=state, v0_bounds=v0_bounds, v1_bounds=v1_bounds, v0_bounds_grad=v0_bounds_grad, v1_bounds_grad=v1_bounds_grad, time=time, sim_time=sim_time, v0_intersection=1, time_halfsteps=time_halfsteps, global_vars=global_vars), 
                            lambda: intersection(i=i, v=vt, v1=params[10], params=params, state=state, v0_bounds=v0_bounds, v1_bounds=v1_bounds, v0_bounds_grad=v0_bounds_grad, v1_bounds_grad=v1_bounds_grad, time=time, sim_time=sim_time, v0_intersection=0, time_halfsteps=time_halfsteps, global_vars=global_vars)
                            )
                        
                        # now both thresholds crossed
                        g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = (0, 0, 1, 1)
                        
                        return new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
    
                    
                    # Check which transition occurs and if needed calculate intersection point
                    vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = jax.lax.cond(
                        d_threshold_gd & (jnp.logical_not(d_threshold_dg)),
                        intersection_case_1,  # Case 1
                        lambda : jax.lax.cond(
                            d_threshold_dg & (jnp.logical_not(d_threshold_gd)), 
                            intersection_case_2,  # Case 2
                            intersection_case_3   # Case 3: Both thresholds crossed
                        )
                    )
                    
                    return vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
                    
                def no_state_change():
                    return vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
                
                vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = jax.lax.cond(
                                (test_threshold_gd < v0_t) & (test_threshold_dg < params[10]),
                                lambda: state_change(vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg),
                                no_state_change
                            )
                    
                
                # if only one condition holds, store which one (at step i)
                # needed for step i+1, when both thresholds are fulfilled, but need 
                # to know which of these conditions was previously false at i
                d_threshold_gd = jnp.where(test_threshold_gd<v0_t, 1, 0)
                d_threshold_dg = jnp.where(test_threshold_dg<params[10], 1, 0)
                    
                return vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
                
                    
        
            # check if transition in state
            vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = jax.lax.cond(
                            state == 0, 
                            lambda: if_glacial(vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg), 
                            lambda: if_deglacial(vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg)
                        )
            
            # after possible intersection, calculate v[i] with 4th order Runge Kutta step
            k1 = Phi(2 * i, vt[i], params, state, global_vars, sim_time, time_steps)
            k2 = Phi(2 * i + 1, vt[i] + k1 * step / 2., params, state, global_vars, sim_time, time_steps)
            k3 = Phi(2 * i + 1, vt[i] + k2 * step / 2., params, state, global_vars, sim_time, time_steps)
            k4 = Phi(2 * i + 2, vt[i] + step * k3, params, state, global_vars, sim_time, time_steps)
            
            vt = vt.at[i + 1].set(vt[i] + step / 6. * (k1 + 2 * k2 + 2 * k3 + k4))
            vt = vt.at[i + 1].set(jnp.where(jnp.isnan(vt[i + 1]), -jnp.inf, vt[i + 1]))
            
            return (vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg)
        
        vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = jax.lax.fori_loop(
                lower=0, upper=time_steps, body_fun=body, init_val=(vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg)
            )
        
        return vt

# no intersections calculated
else:
    # Compute the modelled volume for a set of input parameters using the Runge–Kutta 4th order method
    def modelledVolume(params, global_vars):
        vi = params[11]
        state = 0  # 'g' state represented as 0
        
        params = jnp.array(params)
        global_vars = jnp.array(global_vars)
        Esi = global_vars[0]
        Eco = global_vars[1]
        EnO = global_vars[2]
        
        # total simulation time
        sim_time = jnp.abs(start_year) 
        
        vt = jnp.zeros(time_steps + 1)
        vt = vt.at[0].set(vi)
        step = -start_year/float(time_steps)
        
        def body(i, val):
            vt, state = val
            
            # current time t (full timesteps only)
            t = -1 * start_year - (i * sim_time / time_steps)
            
            # thresholds for state changes (use Esi, Eco, EnO at full time steps only)
            test_threshold_gd = params[6] * Esi[2*i] + params[7] * Eco[2*i] + params[8] * EnO[2*i] + vt[i]
            test_threshold_dg = params[6] * Esi[2*i] + params[7] * Eco[2*i] + params[8] * EnO[2*i]

            # calculate v_0(t)
            v0_t = params[9] - params[12] * t

            # check if transition in state
            def check_glacial():
                return jnp.where((test_threshold_gd > v0_t) & (test_threshold_dg > params[10]), 1, state)

            def check_deglacial():
                return jnp.where((test_threshold_dg < params[10]) & (test_threshold_gd < v0_t), 0, state)

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
    if np.all(np.isfinite(parameters)):
        return 0.0
    else:
        return -np.inf

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
    

        
        
#####################################################################
# Function for calculating the Bayesian Information Criterion (BIC)
# BIC = -2*LogLikelihood + N_Params*ln(N_DataPoints)                             
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
    
else:
    raise ValueError("sea_level_data must be either 'Berends' or 'Rohling'!")


# Load orbital data from Laska
data_orbital = np.loadtxt('../Data/Orbital_Params_-3,6MA-2MA_1kyr_steps.txt')
time = data_orbital[:,0]
mask_interval = np.where(np.logical_and(time>=start_year, 
                                        time<=0))
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
             
    
# Prepare Orbital data for model (normalisation, truncation, interpolation)
# Normalization and truncation of parameters input
Esi = normalise(esinomega)
Eco = normalise(ecosomega)
EnO = normalise(O)

#Interpolation to get data at the time step of 500 years (for half-step Runge-Kutta computation)
Esi = interpol(Esi,2)
Eco = interpol(Eco,2)
EnO = interpol(EnO,2)
time_halfsteps = jnp.array(interpol(time,2))
time = jnp.array(time)

global_vars = np.array([Esi, Eco, EnO])

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

# WalkersIni = np.random.uniform(-1_000, 1_000, size=(nwalkers,ndim))

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

# Nested sampling via dynesty for optimization
elif tuning=='dynesty':
    
    # Define the bounds for each parameter in a dictionary
    bounds = {
        'aEsi': (-10, 10),
        'aEco': (-10, 10),
        'aO': (-10, 10),
        'ag': (-1000, 1000),
        'ad': (-1000, 1000),
        'tau_d': (-1000, 1000),
        'kEsi': (-1000, 1000),
        'kEco': (-1000, 1000),
        'kO': (-1000, 1000),
        'v0': (-200, 200),
        'v1': (-200, 200),
        'vi': (-200, 200),
        'C_v0': (-10, 10),
        'C_taud': (-10, 10),
    }
    
    
    # Prior transformation: gets parameters uniformly drawn from unit cube;
    # Function that transforms samples from the unit cube to the target prior back
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
        
        
    # "Dynamic" nested sampling.
    ndim = len(StartPosition)
    
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
        
    N = len(steps.samples)
    rmses = []
    for i in tqdm(range(N)):
        icevolume = jit_modelledVolume(steps.samples[i,:], global_vars)
        rmse = np.sqrt(np.sum(np.square(icevolume-sea))/len(sea))
        rmses.append(rmse)
    print('\nMinimal RMSE: ',np.min(rmses))
    
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

    bounds = {'lower': [-1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000],
              'upper': [ 1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000]
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
        # params = pm.Uniform('params', lower=bounds['lower'], upper=bounds['upper'], dims='parameters', initval=StartPosition) #, initval=StartPosition
        
        # all params separate
        # sigma_aEsi = pm.HalfNormal('sigma_aEsi', sigma=10)
        # sigma_aEco = pm.HalfNormal('sigma_aEco', sigma=10)
        # sigma_aO = pm.HalfNormal('sigma_aO', sigma=10)
        # sigma_ag = pm.HalfNormal('sigma_ag', sigma=10)
        # sigma_ad = pm.HalfNormal('sigma_ad', sigma=10)
        # sigma_tau_d = pm.HalfNormal('sigma_tau_d', sigma=10)
        # sigma_kEsi = pm.HalfNormal('sigma_kEsi', sigma=10)
        # sigma_kEco = pm.HalfNormal('sigma_kEco', sigma=10)
        # sigma_kO = pm.HalfNormal('sigma_kO', sigma=10)
        # sigma_v0 = pm.HalfNormal('sigma_v0', sigma=10)
        # sigma_v1 = pm.HalfNormal('sigma_v1', sigma=10)
        # sigma_vi = pm.HalfNormal('sigma_vi', sigma=10)
        # sigma_C_v0 = pm.HalfNormal('sigma_C_v0', sigma=10)
        # sigma_C_taud = pm.HalfNormal('sigma_C_taud', sigma=10)
        
        # aEsi = pm.Normal("aEsi", mu=StartPosition_dict['aEsi'], sigma=sigma_aEsi, initval=StartPosition_dict['aEsi']) 
        # aEco = pm.Normal("aEco", mu=StartPosition_dict['aEco'], sigma=sigma_aEco, initval=StartPosition_dict['aEco']) 
        # aO = pm.Normal("aO", mu=StartPosition_dict['aO'], sigma=sigma_aO, initval=StartPosition_dict['aO']) 
        # ag = pm.Normal("ag", mu=StartPosition_dict['ag'], sigma=sigma_ag, initval=StartPosition_dict['ag']) 
        # ad = pm.Normal("ad", mu=StartPosition_dict['ad'], sigma=sigma_ad, initval=StartPosition_dict['ad']) 
        # tau_d = pm.Normal("tau_d", mu=StartPosition_dict['tau_d'], sigma=sigma_tau_d, initval=StartPosition_dict['tau_d']) 
        # kEsi = pm.Normal("kEsi", mu=StartPosition_dict['kEsi'], sigma=sigma_kEsi, initval=StartPosition_dict['kEsi']) 
        # kEco = pm.Normal("kEco", mu=StartPosition_dict['kEco'], sigma=sigma_kEco, initval=StartPosition_dict['kEco'])
        # kO = pm.Normal("kO", mu=StartPosition_dict['kO'], sigma=sigma_kO, initval=StartPosition_dict['kO']) 
        # v0 = pm.Normal("v0", mu=StartPosition_dict['v0'], sigma=sigma_v0, initval=StartPosition_dict['v0']) 
        # v1 = pm.Normal("v1", mu=StartPosition_dict['v1'], sigma=sigma_v1, initval=StartPosition_dict['v1']) 
        # vi = pm.Normal("vi", mu=StartPosition_dict['vi'], sigma=sigma_vi, initval=StartPosition_dict['vi']) 
        # C_v0 = pm.Normal("C_v0", mu=StartPosition_dict['C_v0'], sigma=sigma_C_v0, initval=StartPosition_dict['C_v0']) 
        # C_taud = pm.Normal("C_taud", mu=StartPosition_dict['C_taud'], sigma=sigma_C_taud, initval=StartPosition_dict['C_taud']) 
        
        # aEsi = pm.Uniform("aEsi", lower=-1000, upper=1000, initval=StartPosition_dict['aEsi']) 
        # aEco = pm.Uniform("aEco", lower=-1000, upper=1000, initval=StartPosition_dict['aEco']) 
        # aO = pm.Uniform("aO", lower=-1000, upper=1000, initval=StartPosition_dict['aO']) 
        # ag = pm.Uniform("ag", lower=-1000, upper=1000, initval=StartPosition_dict['ag']) 
        # ad = pm.Uniform("ad", lower=-1000, upper=1000, initval=StartPosition_dict['ad']) 
        # tau_d = pm.Uniform("tau_d", lower=-1000, upper=1000, initval=StartPosition_dict['tau_d']) 
        # kEsi = pm.Uniform("kEsi", lower=-1000, upper=1000, initval=StartPosition_dict['kEsi']) 
        # kEco = pm.Uniform("kEco", lower=-1000, upper=1000, initval=StartPosition_dict['kEco'])
        # kO = pm.Uniform("kO", lower=-1000, upper=1000, initval=StartPosition_dict['kO']) 
        # v0 = pm.Uniform("v0", lower=-1000, upper=1000, initval=StartPosition_dict['v0']) 
        # v1 = pm.Uniform("v1", lower=-1000, upper=1000, initval=StartPosition_dict['v1']) 
        # vi = pm.Uniform("vi", lower=-1000, upper=1000, initval=StartPosition_dict['vi']) 
        # C_v0 = pm.Uniform("C_v0", lower=-1000, upper=1000, initval=StartPosition_dict['C_v0']) 
        # C_taud = pm.Uniform("C_taud", lower=-1000, upper=1000, initval=StartPosition_dict['C_taud']) 
        
        # ps = pt.as_tensor_variable([aEsi, aEco, aO, ag, ad, tau_d, kEsi, kEco, kO, v0, v1, vi, C_v0, C_taud])
        # params = pm.Deterministic('params', ps)

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
# rmse = np.sqrt(np.sum(np.square(icevolume-sea))/len(sea))
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
    ax.set_ylim(np.min([icevolume,sea])-10,np.max([icevolume,sea])+10)
    ax.invert_yaxis()
    ax.set_xlabel("Age (ka)",weight='bold')
    ax.set_ylabel("Ice volume (m sl)",weight='bold')
    
    # plt.savefig('../Plots/GRAD_2.png', dpi=500, bbox_inches='tight')
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
            













