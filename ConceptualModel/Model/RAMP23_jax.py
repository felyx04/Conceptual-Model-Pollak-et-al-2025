#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:15:11 2024

@author: pollakf

Note: - Same os RAMP17_jax.py, but now l instead of beta params
      - kEsi*Esi -> (kEsi+lEsi*v)*Esi
      - New parameterization (Obliquity/Precession accordig to S. Barker) 
        only added for alpha equations, k params untouched!:
          a_Esi * Esi -> a_Esi * Esi
          k_Esi * Esi = (kEsi + lEsi * v) * Esi
          -> 3 new parameters: l_Esi, l_Eco, l_O
      - Switch included to switch between normal and intersection-based model
      - Calculations to spot intersection points at boundary of v(t) 
        and threshold functions
      - PyMC + dynesty tuning added (switch to decide between emcee/ptemcee/PyMC/dynesty)
      - GAPs to verify tuning
      
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

tic = time_module.perf_counter() 


###############################################################################
# BERENDS SEA LEVEL DATA
###############################################################################

###############################################################################
# Berends [-2 Myr - 0]
###############################################################################

# StartPosition = (0.4514824149803438, -0.2361142173854649, 0.9195964672814938, 0.934711136362921, 0.0723176249628068, 5.420825782739939, -17.558966624711957, -8.902062628753015, 0.1872395425019704, 139.60817064278172, -1.91581609696496, 28.436399891914647, 22.65371902812751, 28.38055970017058, 1911.5081506584827, 335.4962157620377, 0.853584824309948, 0.16876460666617277, 0.1403868653190301)
# RMSE = 10.268438365290464

###############################################################################
# Berends [-2.6 Myr - 0]
###############################################################################

StartPosition = (0.3834679918827142, -0.21522894688345673, 0.7967544908165571, 0.9908363954060633, -0.42664821185519486, 5.345427417396081, -10.767187710244116, -6.178225950125125, 0.8188849908257225, 127.3622836218563, -0.8391210167775682, 17.984701880705558, 20.651953809760386, 41.19610088594095, 1977.8274110288676, 454.26909411005107, 0.6750268190905757, 0.1366422196435053, 0.1446407894465409)
# RMSE = 10.022636636439419

###############################################################################
# ROHLING SEA LEVEL DATA
###############################################################################

###############################################################################
# ROHLING [-2 Myr - 0]
###############################################################################

# StartPosition = (0.3876216976531075, -0.10701519642685722, 0.8854716623615354, 0.9905601168011936, 4.043476249105496, 3.806327831464966, 11.04924070591406, 27.758145082071746, 18.505832421104117, 93.96181359756002, -0.7397281780005976, 14.52384115181238, 46.10098960413068, 4.90755308062302, 1616.8690109172153, 964.4707618723633, -0.014796655450864272, -0.2580484531162869, -0.1377379710876507)
# RMSE = 11.32640109501056

###############################################################################
# ROHLING [-2.6 Myr - 0]
###############################################################################

# StartPosition = (0.5186750226185115, -0.13411033571768627, 0.9175331750930464, 0.8236233047095993, 1.5074836937365035, 5.205530308124253, -8.463302499758925, 0.6983595595695533, -6.572363649794056, 111.61022243441153, 2.4362617627885186, 22.084982607797343, 6.184150046091787, 9.812108919217621, 2149.81223952636, 673.7483648149947, 0.41199748562119964, 0.237076919915296, 0.3673187830512461)
# RMSE = 11.090491548501783


# Parameters = [aEsi, aEco, aO, ag, ad, taud0, kEsi, kEco, kO, v0, v1, vi, v0', taud0', t2, t1, lEsi, lEco, lO]
#                0      1    2   3   4     5     6     7    8   9  10  11  12     13    14  15   16,   17,  18

parameter_names = ['aEsi', 'aEco', 'aO', 'ag', 'ad', 'taud0', 'kEsi', 'kEco', 'kO', 'v0', 'v1', 'vi', 'v0_prime', 'taud0_prime', 't2', 't1', 
                   'lEsi', 'lEco', 'lO']
   
#Number of walkers (verifying ; nwalkers > 2 * number of parameters)
nwalkers = 100

#Number of iterations 
niterations = 50_000   #100_000   (for 1000yr res: 500_000 + walker_jumps=0.3)

#Define the first position of each walkers relatively to StartPosition. When walkers_jump is high, walkers are far from StartPosition. 
walkers_jump = 0.01

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
# gap = (-int(1_200), -int(700)) 
# gap = (-int(130), -int(0)) 
# gap = (-int(2600), -int(700))
gap = None

time_steps = int(-start_year*1e3/resolution)  # number of timesteps in model

# Select tuning procedure
tuning = 'emcee'  # Options: 'emcee', 'ptemcee', 'pymc', 'dynesty'
dynesty_static = True  # static nested sampling (keep True)

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
    

    # t>t2: Before Ramp
    def before_ramp():
        tau_t = jnp.where(params[13] == 0, 1e-8, params[13])
        return tau_t

    # t2 >= t >= t1: Ramp
    def during_ramp():
        tau_t = params[5] - ((params[5] - params[13]) / jnp.abs(params[14] - params[15])) * jnp.abs(t - params[15])
        tau_t = jnp.where(tau_t == 0, 1e-8, tau_t)
        return tau_t

    # t1 > t: After Ramp
    def after_ramp():
        tau_t = jnp.where(params[5] == 0, 1e-8, params[5])
        return tau_t

    tau_t = jax.lax.cond(t > params[14], before_ramp, 
                               lambda: jax.lax.cond(t>=params[15] , during_ramp, after_ramp))

    # Calculate change in ice volume dv/dt
    dvdt = jnp.where(state == 0,
                     -params[0] * Esi[i] - params[1] * Eco[i] - params[2] * EnO[i] + params[3],
                     -params[0] * Esi[i] - params[1] * Eco[i] - params[2] * EnO[i] + params[4] - vt / tau_t)
    
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
    def calc_bounds(params, global_vars, time_steps, vt):
        Esi = global_vars[0]
        Eco = global_vars[1]
        EnO = global_vars[2]
        
        v0_bounds = jnp.zeros(time_steps+1)
        v1_bounds = jnp.zeros(time_steps+1)
        
        # complete simulation time
        sim_time = jnp.abs(start_year)
        
        # body function for JAX loop
        def body_fn(i, val):
            v0_bounds, v1_bounds = val
            
            # current time t
            t = -1 * start_year - (i * sim_time / time_steps)
            
            # t>t2: Before Ramp
            def before_ramp():
                v0_t = params[12]
                return v0_t

            # t2 >= t >= t1: Ramp
            def during_ramp():
                v0_t = params[9] - ((params[9] - params[12]) / jnp.abs(params[14] - params[15])) * jnp.abs(t - params[15])
                return v0_t

            # t1 > t: After Ramp
            def after_ramp():
                v0_t = params[9]
                return v0_t
            
            # Compute v0_t at time step i
            v0_t = jax.lax.cond(t > params[14], before_ramp, 
                                       lambda: jax.lax.cond(t>=params[15] , during_ramp, after_ramp))
            
            # Update bounds
            v0_bounds = v0_bounds.at[i].set(v0_t - (params[6]+params[16]*vt[i]) * Esi[2*i] - (params[7]+params[17]*vt[i]) * Eco[2*i] - (params[8]+params[18]*vt[i]) * EnO[2*i])
            v1_bounds = v1_bounds.at[i].set((params[6]+params[16]*vt[i]) * Esi[2*i] + (params[7]+params[17]*vt[i]) * Eco[2*i] + (params[8]+params[18]*vt[i]) * EnO[2*i])
            
            return v0_bounds, v1_bounds
        

        # Use JAX loop to handle iteration
        v0_bounds, v1_bounds = jax.lax.fori_loop(lower=0, upper=time_steps+1, body_fun=body_fn, init_val=(v0_bounds, v1_bounds))
        
        return v0_bounds, v1_bounds
    
    #####################################################################
    # Function to calculate intersection points between v(t) and deglac. threshold (based on v0)
    def calc_intersect_v0(i, v, time, params, global_vars):
        v_grad = np_gradient(v)
        
        v0_bounds, v1_bounds = calc_bounds(params, global_vars, time_steps, v)
        
        v0_bounds_grad = np_gradient(v0_bounds)
        
        # calculate intersection
        intersect = -(v[i-1]-v0_bounds[i-1])/(v_grad[i-1]-v0_bounds_grad[i-1])
        
        return intersect, time[i-1]-intersect


    #####################################################################
    # Function to calculate intersection points between insolation and v1
    def calc_intersect_v1(i, v, v1, time, params, global_vars):
        v0_bounds, v1_bounds = calc_bounds(params, global_vars, time_steps, v)
        
        v1_bounds_grad = np_gradient(v1_bounds)
        
        # calculate intersection
        intersect = (v1-v1_bounds[i-1]) / v1_bounds_grad[i-1]
        
        return intersect, time[i-1]-intersect


    #####################################################################
    # Function to calculate v(i+t') at intersection point and recalculate v(i) after intersection point
    def intersection(i, v, v1, state, params, time, sim_time, v0_intersection, time_halfsteps, global_vars):
        # Conditional intersection calculation based on v0_intersection
        intersect, time_intersect = jax.lax.cond(
            v0_intersection,
            lambda: calc_intersect_v0(i, v, time, params, global_vars),    # calculate intersection point between v(t) and degl. threshold (based on v0)
            lambda: calc_intersect_v1(i, v, v1, time, params, global_vars) # calculate intersection point between insolation and v1
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
        
        # t>t2: Before Ramp
        def before_ramp():
            tau_t = jnp.where(params[13] == 0, 1e-8, params[13])
            return tau_t
    
        # t2 >= t >= t1: Ramp
        def during_ramp():
            tau_t = params[5] - ((params[5] - params[13]) / jnp.abs(params[14] - params[15])) * jnp.abs(time_intersect - params[15])
            tau_t = jnp.where(tau_t == 0, 1e-8, tau_t)
            return tau_t
    
        # t1 > t: After Ramp
        def after_ramp():
            tau_t = jnp.where(params[5] == 0, 1e-8, params[5])
            return tau_t
    
        tau_t = jax.lax.cond(time_intersect > params[14], before_ramp, 
                                   lambda: jax.lax.cond(time_intersect>=params[15] , during_ramp, after_ramp))
    
        # calculate Esi, Eco and EnO at intersection point
        Esi_intersect = jnp.interp(time_intersect, time_halfsteps[::-1], Esi[::-1])
        Eco_intersect = jnp.interp(time_intersect, time_halfsteps[::-1], Eco[::-1])
        EnO_intersect = jnp.interp(time_intersect, time_halfsteps[::-1], EnO[::-1])
        
        # Calculate change in ice volume dv/dt
        dvdt = jnp.where(state == 0,
                         -params[0] * Esi_intersect - params[1] * Eco_intersect - params[2] * EnO_intersect + params[3],
                         -params[0] * Esi_intersect - params[1] * Eco_intersect - params[2] * EnO_intersect + params[4] - vt/tau_t)
        
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
            test_threshold_gd = (params[6]+params[16]*vt[i]) * Esi[2*i] + (params[7]+params[17]*vt[i]) * Eco[2*i] + (params[8]+params[18]*vt[i]) * EnO[2*i] + vt[i]
            test_threshold_dg = (params[6]+params[16]*vt[i]) * Esi[2*i] + (params[7]+params[17]*vt[i]) * Eco[2*i] + (params[8]+params[18]*vt[i]) * EnO[2*i]
    
            # t>t2: Before Ramp
            def before_ramp():
                v0_t = params[12]
                return v0_t
    
            # t2 >= t >= t1: Ramp
            def during_ramp():
                v0_t = params[9] - ((params[9] - params[12]) / jnp.abs(params[14] - params[15])) * jnp.abs(t - params[15])
                return v0_t
    
            # t1 > t: After Ramp
            def after_ramp():
                v0_t = params[9]
                return v0_t
    
            v0_t = jax.lax.cond(t > params[14], before_ramp, 
                                       lambda: jax.lax.cond(t>=params[15] , during_ramp, after_ramp))
            
            
            # check for state change and interesction in glacial state 
            def if_glacial(vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg):
                def state_change(vt, state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg):
                    # only calculate intersections, if v(t) crossed gd_threshold, but insolation> v1 was already fulfilled at last timestep
                    def intersection_case_1():
                        new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=vt, v1=params[10], params=params, state=state, time=time, sim_time=sim_time, v0_intersection=1, time_halfsteps=time_halfsteps, global_vars=global_vars)    
                        return new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
                    
                    # only calculate intersections, if insolation crossed v1 threshold, but v(t)<gd_threshold was already fulfilled at last timestep
                    def intersection_case_2():
                        new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=vt, v1=params[10], params=params, state=state, time=time, sim_time=sim_time, v0_intersection=0, time_halfsteps=time_halfsteps, global_vars=global_vars)
                        return new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
                        
                    # if both thresholds were crossed at same timestep
                    def intersection_case_3():
                        # calculate both intersection points
                        intersect_v0, time_v0 = calc_intersect_v0(i, vt, time, params, global_vars) 
                        intersect_v1, time_v1 = calc_intersect_v1(i, vt, params[10], time, params, global_vars) 
                        
                        # choose the later (=smaller) intersection point
                        new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = jax.lax.cond(
                            time_v0 <= time_v1, 
                            lambda: intersection(i=i, v=vt, v1=params[10], params=params, state=state, time=time, sim_time=sim_time, v0_intersection=1, time_halfsteps=time_halfsteps, global_vars=global_vars), 
                            lambda: intersection(i=i, v=vt, v1=params[10], params=params, state=state, time=time, sim_time=sim_time, v0_intersection=0, time_halfsteps=time_halfsteps, global_vars=global_vars)
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
                        new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=vt, v1=params[10], params=params, state=state, time=time, sim_time=sim_time, v0_intersection=0, time_halfsteps=time_halfsteps, global_vars=global_vars)
                        return new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
                    
                    # only calculate intersections, if v(t) gets smaller than degl. threshold, but insolation was already smaller than v1 at last timestep
                    def intersection_case_2():
                        new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=vt, v1=params[10], params=params, state=state, time=time, sim_time=sim_time, v0_intersection=1, time_halfsteps=time_halfsteps, global_vars=global_vars)        
                        return new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg
                        
                    # if both thresholds were crossed at same timestep
                    def intersection_case_3():
                        # calculate both intersection points
                        intersect_v0, time_v0 = calc_intersect_v0(i, vt, time, params, global_vars) 
                        intersect_v1, time_v1 = calc_intersect_v1(i, vt, params[10], time, params, global_vars)
                        
                        # choose the later (=smaller) intersection point
                        new_v, new_state, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = jax.lax.cond(
                            time_v0 <= time_v1, 
                            lambda: intersection(i=i, v=vt, v1=params[10], params=params, state=state, time=time, sim_time=sim_time, v0_intersection=1, time_halfsteps=time_halfsteps, global_vars=global_vars), 
                            lambda: intersection(i=i, v=vt, v1=params[10], params=params, state=state, time=time, sim_time=sim_time, v0_intersection=0, time_halfsteps=time_halfsteps, global_vars=global_vars)
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
            test_threshold_gd = (params[6]+params[16]*vt[i]) * Esi[2*i] + (params[7]+params[17]*vt[i]) * Eco[2*i] + (params[8]+params[18]*vt[i]) * EnO[2*i] + vt[i]
            test_threshold_dg = (params[6]+params[16]*vt[i]) * Esi[2*i] + (params[7]+params[17]*vt[i]) * Eco[2*i] + (params[8]+params[18]*vt[i]) * EnO[2*i]

            # t>t2: Before Ramp
            def before_ramp():
                v0_t = params[12]
                return v0_t

            # t2 >= t >= t1: Ramp
            def during_ramp():
                v0_t = params[9] - ((params[9] - params[12]) / jnp.abs(params[14] - params[15])) * jnp.abs(t - params[15])
                return v0_t

            # t1 > t: After Ramp
            def after_ramp():
                v0_t = params[9]
                return v0_t

            v0_t = jax.lax.cond(t > params[14], before_ramp, 
                                       lambda: jax.lax.cond(t>=params[15] , during_ramp, after_ramp))

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
    aEsi, aEco, aO, ag, ad, taud0, kEsi, kEco, kO, v0, v1, vi, v0_prime, taud0_prime, t2, t1, lEsi, lEco, lO = parameters
    # if (-10.0 < aEsi < 10.0) and (-10.0 < aEco < 10.0) and (-10.0 < aO < 10.0)  and (-10.0 < ag < 10.0) and (-10.0 < ad < 10.0) and (-30.0 < taud0 < 30.0) and (-30.0 < kEsi < 30.0) and (-50.0 < kEco < 50.0) and (-30.0 < kO < 30.0) and (50.0 < v0 < 200.0) and (-50.0 < v1 < 50.0) and (-50.0 < vi < 50.0) and (-30.0 < v0_prime < 30.0) and (t1 < t2 < -start_year) and (0 < t1 < t2):   
    if (t1 < t2 < -start_year) and (0 < t1 < t2) and np.all(np.isfinite(parameters)) and np.all(np.array(parameters)>=-1e5) and np.all(np.array(parameters)<=1e5):
        return 0.0
    
    # (-1 < lEsi < 1) and (-1 < lEco < 1) and (-1 < lO < 1) 
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
esinomega = -1*data_orbital[:,1][mask_interval]#[::-1]
ecosomega = -1*data_orbital[:,2][mask_interval]#[::-1]
O = data_orbital[:,3][mask_interval]#[::-1]
    

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
        'aEsi': (-10, 10),
        'aEco': (-10, 10),
        'aO': (-10, 10),
        'ag': (-1000, 1000),
        'ad': (-1000, 1000),
        'taud0': (-1000, 1000),
        'kEsi': (-1000, 1000),
        'kEco': (-1000, 1000),
        'kO': (-1000, 1000),
        'v0': (-200, 200),
        'v1': (-200, 200),
        'vi': (-200, 200),
        'v0_prime': (-200, 200),
        'taud0_prime': (-1000, 1000),
        't2': (0, -start_year),
        't1': (0, -start_year),
        'lEsi': (-100, 100),
        'lEco': (-100, 100),
        'lO': (-100, 100)
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

    bounds = {'lower': [-1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000, -1_000,  1_001,       0,    -1_000, -1_000, -1_000],
              'upper': [ 1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000,  1_000, -start_year, 1_000,  1_000,  1_000,  1_000]
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
        # sigma_taud0 = pm.HalfNormal('sigma_taud0', sigma=10)
        # sigma_kEsi = pm.HalfNormal('sigma_kEsi', sigma=10)
        # sigma_kEco = pm.HalfNormal('sigma_kEco', sigma=10)
        # sigma_kO = pm.HalfNormal('sigma_kO', sigma=10)
        # sigma_v0 = pm.HalfNormal('sigma_v0', sigma=10)
        # sigma_v1 = pm.HalfNormal('sigma_v1', sigma=10)
        # sigma_vi = pm.HalfNormal('sigma_vi', sigma=10)
        # sigma_v0_prime = pm.HalfNormal('sigma_v0_prime', sigma=10)
        # sigma_taud0_prime = pm.HalfNormal('sigma_taud0_prime', sigma=10)
        # sigma_t2 = pm.HalfNormal('sigma_t2', sigma=100)
        # sigma_t1 = pm.HalfNormal('sigma_t1', sigma=100)
        # sigma_lEsi = pm.HalfNormal('sigma_lEsi', sigma=10)
        # sigma_lEco = pm.HalfNormal('sigma_lEco', sigma=10)
        # sigma_lO = pm.HalfNormal('sigma_lO', sigma=10)
        
        # aEsi = pm.Normal("aEsi", mu=StartPosition_dict['aEsi'], sigma=sigma_aEsi, initval=StartPosition_dict['aEsi']) 
        # aEco = pm.Normal("aEco", mu=StartPosition_dict['aEco'], sigma=sigma_aEco, initval=StartPosition_dict['aEco']) 
        # aO = pm.Normal("aO", mu=StartPosition_dict['aO'], sigma=sigma_aO, initval=StartPosition_dict['aO']) 
        # ag = pm.Normal("ag", mu=StartPosition_dict['ag'], sigma=sigma_ag, initval=StartPosition_dict['ag']) 
        # ad = pm.Normal("ad", mu=StartPosition_dict['ad'], sigma=sigma_ad, initval=StartPosition_dict['ad']) 
        # taud0 = pm.Normal("taud0", mu=StartPosition_dict['taud0'], sigma=sigma_taud0, initval=StartPosition_dict['taud0']) 
        # kEsi = pm.Normal("kEsi", mu=StartPosition_dict['kEsi'], sigma=sigma_kEsi, initval=StartPosition_dict['kEsi']) 
        # kEco = pm.Normal("kEco", mu=StartPosition_dict['kEco'], sigma=sigma_kEco, initval=StartPosition_dict['kEco'])
        # kO = pm.Normal("kO", mu=StartPosition_dict['kO'], sigma=sigma_kO, initval=StartPosition_dict['kO']) 
        # v0 = pm.Normal("v0", mu=StartPosition_dict['v0'], sigma=sigma_v0, initval=StartPosition_dict['v0']) 
        # v1 = pm.Normal("v1", mu=StartPosition_dict['v1'], sigma=sigma_v1, initval=StartPosition_dict['v1']) 
        # vi = pm.Normal("vi", mu=StartPosition_dict['vi'], sigma=sigma_vi, initval=StartPosition_dict['vi']) 
        # v0_prime = pm.Normal("v0_prime", mu=StartPosition_dict['v0_prime'], sigma=sigma_v0_prime, initval=StartPosition_dict['v0_prime']) 
        # taud0_prime = pm.Normal("taud0_prime", mu=StartPosition_dict['taud0_prime'], sigma=sigma_taud0_prime, initval=StartPosition_dict['taud0_prime']) 
        # t2 = pm.Normal("t2", mu=StartPosition_dict['t2'], sigma=sigma_t2, initval=StartPosition_dict['t2']) 
        # t1 = pm.Normal("t1", mu=StartPosition_dict['t1'], sigma=sigma_t1, initval=StartPosition_dict['t1'])  
        # lEsi = pm.Normal("lEsi", mu=StartPosition_dict['lEsi'], sigma=sigma_lEsi, initval=StartPosition_dict['lEsi']) 
        # lEco = pm.Normal("lEco", mu=StartPosition_dict['lEco'], sigma=sigma_lEco, initval=StartPosition_dict['lEco']) 
        # lO = pm.Normal("lO", mu=StartPosition_dict['lO'], sigma=sigma_lO, initval=StartPosition_dict['lO']) 
        
        # aEsi = pm.Uniform("aEsi", lower=-1000, upper=1000, initval=StartPosition_dict['aEsi']) 
        # aEco = pm.Uniform("aEco", lower=-1000, upper=1000, initval=StartPosition_dict['aEco']) 
        # aO = pm.Uniform("aO", lower=-1000, upper=1000, initval=StartPosition_dict['aO']) 
        # ag = pm.Uniform("ag", lower=-1000, upper=1000, initval=StartPosition_dict['ag']) 
        # ad = pm.Uniform("ad", lower=-1000, upper=1000, initval=StartPosition_dict['ad']) 
        # taud0 = pm.Uniform("taud0", lower=-1000, upper=1000, initval=StartPosition_dict['taud0']) 
        # kEsi = pm.Uniform("kEsi", lower=-1000, upper=1000, initval=StartPosition_dict['kEsi']) 
        # kEco = pm.Uniform("kEco", lower=-1000, upper=1000, initval=StartPosition_dict['kEco'])
        # kO = pm.Uniform("kO", lower=-1000, upper=1000, initval=StartPosition_dict['kO']) 
        # v0 = pm.Uniform("v0", lower=-1000, upper=1000, initval=StartPosition_dict['v0']) 
        # v1 = pm.Uniform("v1", lower=-1000, upper=1000, initval=StartPosition_dict['v1']) 
        # vi = pm.Uniform("vi", lower=-1000, upper=1000, initval=StartPosition_dict['vi']) 
        # v0_prime = pm.Uniform("v0_prime", lower=-1000, upper=1000, initval=StartPosition_dict['v0_prime']) 
        # taud0_prime = pm.Uniform("taud0_prime", lower=-1000, upper=1000, initval=StartPosition_dict['taud0_prime']) 
        # t2 = pm.Uniform("t2", lower=1001, upper=-start_year, initval=StartPosition_dict['t2']) 
        # t1 = pm.Uniform("t1", lower=0, upper=1000, initval=StartPosition_dict['t1'])  
        # lEsi = pm.Uniform("lEsi", lower=-1000, upper=1000, initval=StartPosition_dict['lEsi']) 
        # lEco = pm.Uniform("lEco", lower=-1000, upper=1000, initval=StartPosition_dict['lEco']) 
        # lO = pm.Uniform("lO", lower=-1000, upper=1000, initval=StartPosition_dict['lO']) 
        
        # ps = pt.as_tensor_variable([aEsi, aEco, aO, ag, ad, taud0, kEsi, kEco, kO, v0, v1, vi, v0_prime, taud0_prime, t2, t1, lEsi, lEco, lO])
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
    
    # plt.savefig('../Figures/RAMP_23.png', dpi=500, bbox_inches='tight')
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
            













