#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:39:23 2024

@author: pollakf

Note: - Based on RAMP5_jax.py
      - Switch for intersections added (if False: equal to RAMP5, but also with gaps)
      - New calculations added to spot intersection points at boundary of v(t) 
        and threshold functions
      - Added GAPs to verify tuning
      - CURRENT STANDARD MODEL!
      
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
# Berends [-2 Myr - 0]
###############################################################################

# Parameters from Pollak et al. (2025)
# StartPosition = (0.07373640324372488, -0.24092901923688892, 0.6371457643410627, 0.8774272529900875, 1.0373504062615417, 4.708218737703654, 21.805157984193343, -2.5331559193820468, 14.255078909083402, 110.83571072406629, -5.388492422749707, 18.352854167428056, 24.96187472479393, 13.518454850507398, 1995.2653473716073, 687.9695915111181)
# RMSE = 11.593961988822706

# Improved parameters
# StartPosition = (-0.1621092175418255, -0.4669094147420374, 0.6495209595656153, 0.9140309110657654, 0.8724933457114048, 5.44492938684391, 21.12271450975811, 2.618105205309803, 13.479298402811764, 109.93607946462964, -1.9554275662144107, 15.662173791012748, 31.037637088130765, 13.444840632431692, 1887.8699396985578, 673.079496250923)
# RMSE = 11.570750239736304

###############################################################################
# Berends [-2.6 Myr - 0]
###############################################################################

# Parameters from Pollak et al. (2025)
StartPosition = (0.09235458604990571, -0.14460424678965283, 0.587063356079243, 0.8899639945686884, 0.1647052467728392, 6.337560462113291, 17.69686418811786, 1.4040939575987441, 11.570672119241756, 98.38769322382691, 2.3544085304028566, 9.176650804487933, -12.916000589465284, 31.417498351214654, 2159.346069989433, 857.9325641775804)
# RMSE = 11.3842691256649

# Improved parameters
# StartPosition = (0.03835904843788285, -0.24321776128757833, 0.6286290332326061, 0.9336119560870202, 0.11778286892024425, 5.967762970808394, 18.443418207226387, 1.814521520147082, 12.235594936802272, 101.27310409355096, 2.275814869432395, 23.908784323971645, -16.03118737245265, 33.01613217972704, 2173.7165852023236, 860.2654535905915)
# RMSE = 11.302898305369384


###############################################################################
# Berends [-3.599 Myr - 0]
###############################################################################

# StartPosition = (-0.20416973684716844, -0.2317187655870222, 0.5613706777422738, 0.9749948715441965, -0.11316705515366085, 6.492854034041283, 20.030008682547688, 3.0123955095105517, 13.291462410129265, 104.76724990680347, -2.2834047724817443, -19.60974028977521, -91.54809048106617, 56.8090595498631, 3478.7456636103925, 840.6025359328102)
# RMSE = 10.688418576509187


###############################################################################
# ROHLING SEA LEVEL DATA
###############################################################################

###############################################################################
# ROHLING [-2 Myr - 0]
###############################################################################

# StartPosition = (0.7115867779377646, 0.22193020864590152, 0.9235955426547092, 0.8239045342862367, 1.7595539513351408, 6.583577462928954, 12.258180834138372, 13.80825807820386, 9.355435623605594, 99.29165684969628, -2.626906191071508, 14.899458813146921, 44.04780543543359, 10.50791129018398, 1680.4679164623665, 821.4298128459351)
# RMSE = 11.831951812254456


###############################################################################
# ROHLING [-2.6 Myr - 0]
###############################################################################

# StartPosition = (0.4839717127691969, 0.1631926211482264, 0.8726544514464614, 0.8423701125786861, 2.730500079900103, 5.2967978778496025, 14.631652219401442, 14.22447658213209, 11.080590572667738, 102.01888605460721, 0.3803568921545758, 26.776414482325915, -4.9408058754487785, 6.459059205345467, 2287.263577174279, 878.2970760789136)
# RMSE = 11.699645317799451


###############################################################################
# ROHLING [-3.599 Myr - 0]
###############################################################################

# StartPosition = (0.34709005446834, 0.05615790276590549, 0.8706196986839485, 0.9240550606377803, 5.460969794421124, 4.3073781546923176, 14.761914136550011, 13.840557699390901, 10.626302235554434, 101.84947893804072, 0.8519809050260048, -10.580931012088095, -78.98844006886877, 0.3590472301683907, 3397.850464862552, 906.3465109010492)
# RMSE = 11.276158209785077



# Parameters = [aEsi, aEco, aO, ag, ad, taud0, kEsi, kEco, kO, v0, v1, vi, v0', taud0', t2, t1]
#                0      1    2   3   4     5     6     7    8   9  10  11  12     13    14  15
   
parameter_names = ['aEsi', 'aEco', 'aO', 'ag', 'ad', 'taud0', 'kEsi', 'kEco', 'kO', 'v0', 'v1', 'vi', 'v0_prime', 'taud0_prime', 't2', 't1']


# Set resolution of model
resolution = 1000

# Start year in kyr 
start_year = -int(2_600)

# Future simulation years in kyr (between 0 and 1_000)
future_time = int(250)

# Gap included for tuning: Model is not tuned during this time interval. gap=(start_gap[kyr BP], end_gap[kyr BP])
# gap = (-int(1_200), -int(700)) 
# gap = (-int(130), -int(0)) 
gap = None

# switch for intersections 
intersections = False

# Set title for Plot
title = 'RAMP experiment'

# Set sea-level data: either Berends or Rohling (LR04 based + tuned age) 
sea_level_data = 'Berends'

# save data in binary file with key:value pairs
# save_data = '../Data/RAMP7_Rohling_2,6Myr.pkl'
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
    
    # full simulation time
    sim_time = abs(start_year)+abs(future_time)
    
    # current time t
    t = -1*start_year-((i*sim_time/time_steps)/2) 
    
    # t>t2: Before Ramp
    if t>t2:
        tau_t = taud0_prime 
        
    # t2>=t>=t1: Ramp
    elif t2>=t>=t1:
        # tau(t) = taud0 - (taud0-taud0')/(t2-t1) * (t-t1)
        tau_t = taud0 - ((taud0-taud0_prime)/np.abs(t2-t1)) * np.abs(t-t1)    
    
    # t1>t: After Ramp
    else:
        tau_t = taud0
            
    if S[0] == "g" :
        dvdt = -aEsi*Esi[i]-aEco*Eco[i]-aO*EnO[i]+ag
   
    else :
        dvdt = -aEsi*Esi[i]-aEco*Eco[i]-aO*EnO[i]+ad-v/tau_t     
    
    return dvdt

# Calculate intersections
if intersections:
    #####################################################################
    # Function to calculate intersection points between v(t) and deglac. threshold (based on v0)
    def calc_intersect_v0(i, v, v0_bounds, time):
        # calculate gradient, including v_new
        v_grad = np.gradient(v, resolution*1e-3)
        v0_bounds = v0_bounds[::2]
        v0_bounds_grad = np.gradient(v0_bounds, resolution*1e-3)
        
        # calculate intersection
        intersect = -(v[i-1]-v0_bounds[i-1])/(v_grad[i-1]-v0_bounds_grad[i-1])
        
        return intersect, time[i-1]-intersect

    #####################################################################
    # Function to calculate intersection points between insolation and v1
    def calc_intersect_v1(i, v1, v1_bounds, time):
        # calculate gradient
        v1_bounds = v1_bounds[::2]
        v1_bounds_grad = np.gradient(v1_bounds, resolution*1e-3)
        
        # calculate intersection
        intersect = (v1-v1_bounds[i-1]) / v1_bounds_grad[i-1]
        
        return intersect, time[i-1]-intersect


    #####################################################################
    # Function to calculate v(i+t') at intersection point and recalculate v(i) after intersection point
    def intersection(i, v, v1, v0_bounds, v1_bounds, time, intersections, time_intersections, v_intersections, vs_old, times_old, sim_time, v0_intersection, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg):
        if v0_intersection:
            # calculate intersection point between v(t) and degl. threshold (based on v0)
            intersect, time_intersect = calc_intersect_v0(i, v, v0_bounds, time)
        else:
            # calculate intersection point between insolation and v1
            intersect, time_intersect = calc_intersect_v1(i, v1, v1_bounds, time)
            
        intersections.append(intersect)
        time_intersections.append(time_intersect)
        
        # calculate v at t+t' (intersection point)
        j = i-1
        k1 = Phi(2*j,v[j])
        k2 = Phi_intersect(v=v[j]+k1*intersect/2, time_intersect=time[j]-intersect/2)
        k3 = Phi_intersect(v=v[j]+k2*intersect/2, time_intersect=time[j]-intersect/2)
        k4 = Phi_intersect(v=v[j]+k3*intersect, time_intersect=time_intersect)
        v_intersect = v[j] + intersect/6.*(k1+2*k2+2*k3+k4)
        v_intersections.append(v_intersect)
        
        if S[0]=='g':
            # change state
            S[0] = "d"
            S[1] = i
        else:
            # change state
            S[0] = "g"
            Term_duration = (i-S[1])*sim_time/time_steps #Compute the duration of a termination
            Term_start = (abs(start_year)-S[1])*sim_time/time_steps #Compute the start of a termination
            ListDuration.append(Term_duration)
            ListStart.append(Term_start)
        
        # store deprecated v[i]
        vs_old.append(v[i])
        times_old.append(time[i])
        
        # recalculate v at full position i (t+step). In "d" state 
        step_after_intersect = abs(time[i]-time_intersect)  # time step between t_intersect and t[i] (after intersection)
        k1 = Phi_intersect(v=v_intersect, time_intersect=time_intersect)
        k2 = Phi_intersect(v=v_intersect+k1*step_after_intersect/2, time_intersect=time_intersect-step_after_intersect/2)
        k3 = Phi_intersect(v=v_intersect+k2*step_after_intersect/2, time_intersect=time_intersect-step_after_intersect/2)
        k4 = Phi(2*i, v_intersect+step_after_intersect*k3)
        v[i] = v_intersect + step_after_intersect/6.*(k1+2*k2+2*k3+k4)
        
        # now both thresholds crossed
        if v0_intersection:
            g_threshold_gd = True
            g_threshold_dg = True
            d_threshold_gd = False
            d_threshold_dg = False
        else:
            d_threshold_gd = True
            d_threshold_dg = True
            g_threshold_gd = False
            g_threshold_dg = False
            
        return v, intersections, time_intersections, v_intersections, vs_old, times_old, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg


    #######################################################
    #Compute the derivative at intersection points
    def Phi_intersect(v, time_intersect):
        
        # time_intersect>t2: Before Ramp
        if time_intersect>t2:
            tau_t = taud0_prime 
            
        # t2>=time_intersect>=t1: Ramp
        elif t2>=time_intersect>=t1:
            # tau(t) = taud0 - (taud0-taud0')/(t2-t1) * (time_intersect-t1)
            tau_t = taud0 - ((taud0-taud0_prime)/np.abs(t2-t1)) * np.abs(time_intersect-t1)    
        
        # t1>time_intersect: After Ramp
        else:
            tau_t = taud0
            
        # calculate Esi, Eco and EnO at intersection point
        Esi_intersect = np.interp(time_intersect, np.flip(time_halfsteps), np.flip(Esi))
        Eco_intersect = np.interp(time_intersect, np.flip(time_halfsteps), np.flip(Eco))
        EnO_intersect = np.interp(time_intersect, np.flip(time_halfsteps), np.flip(EnO))
                
        if S[0] == "g" :
            dvdt = -aEsi*Esi_intersect-aEco*Eco_intersect-aO*EnO_intersect+ag
       
        else :
            dvdt = -aEsi*Esi_intersect-aEco*Eco_intersect-aO*EnO_intersect+ad-v/tau_t     
        
        return dvdt
    
    ##########################################################
    #Compute the modelled volume for the best parameters using the Runge–Kutta 4th order method
    def modelledVolume(start_year, future_time, vi, n) :
        v = np.zeros(n+1)
        v[0] = vi
        state.append(S[1])
        intersections = []
        time_intersections = []
        v_intersections = []
        vs_old = []
        times_old = []
        g_threshold_gd = False
        g_threshold_dg = False
        d_threshold_gd = False
        d_threshold_dg = False
        
        # full simulation time
        sim_time = abs(start_year)+abs(future_time)
        
        step = (future_time-start_year)/float(n)
        print('Step: ', step)
        for i in range(n):
            # current time t
            t = -1*start_year-(i*sim_time/time_steps)
            
            # thresholds for state changes (use Esi, Eco, EnO at full time steps only)
            test_threshold_gd = kEsi*Esi[2*i]+kEco*Eco[2*i]+kO*EnO[2*i]+v[i]
            test_threshold_dg = kEsi*Esi[2*i]+kEco*Eco[2*i]+kO*EnO[2*i]     
            
            # t>t2: Before Ramp
            if t>t2:
                v0_t = v0_prime
                
            # t2>=t>=t1: Ramp
            elif t2>=t>=t1:
                # v0(t) = v0 - (v0-v0')/(t2-t1) * (t-t1)
                v0_t = v0 - ((v0-v0_prime)/np.abs(t2-t1)) * np.abs(t-t1) 
            
            # t1>t: After Ramp
            else:
                v0_t = v0
            
            if S[0] == "g":
                # state change
                if test_threshold_gd>v0_t and test_threshold_dg>v1:
                    # only calculate intersections, if v(t) crossed gd_threshold, but insolation> v1 was already fulfilled at last timestep
                    if g_threshold_dg and not g_threshold_gd:
                        v, intersections, time_intersections, v_intersections, vs_old, times_old, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=v, v1=v1, v0_bounds=v0_bounds, v1_bounds=v1_bounds, time=time, intersections=intersections, time_intersections=time_intersections, v_intersections=v_intersections, vs_old=vs_old, times_old=times_old, sim_time=sim_time, v0_intersection=True, g_threshold_gd=g_threshold_gd, g_threshold_dg=g_threshold_dg, d_threshold_gd=d_threshold_gd, d_threshold_dg=d_threshold_dg)
                    
                    # only calculate intersections, if insolation crossed v1 threshold, but v(t)<gd_threshold was already fulfilled at last timestep
                    elif g_threshold_gd and not g_threshold_dg:
                        v, intersections, time_intersections, v_intersections, vs_old, times_old, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=v, v1=v1, v0_bounds=v0_bounds, v1_bounds=v1_bounds, time=time, intersections=intersections, time_intersections=time_intersections, v_intersections=v_intersections, vs_old=vs_old, times_old=times_old, sim_time=sim_time, v0_intersection=False, g_threshold_gd=g_threshold_gd, g_threshold_dg=g_threshold_dg, d_threshold_gd=d_threshold_gd, d_threshold_dg=d_threshold_dg)
                    
                    # if both thresholds were crossed at same timestep
                    else: 
                        # calculate both intersection points
                        intersect_v0, time_v0 = calc_intersect_v0(i, v, v0_bounds, time)
                        intersect_v1, time_v1 = calc_intersect_v1(i, v1, v1_bounds, time)
                        
                        # choose the later (=smaller) intersection point
                        if time_v0 <= time_v1:   # v0 threshold is crossed last
                            v, intersections, time_intersections, v_intersections, vs_old, times_old, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=v, v1=v1, v0_bounds=v0_bounds, v1_bounds=v1_bounds, time=time, intersections=intersections, time_intersections=time_intersections, v_intersections=v_intersections, vs_old=vs_old, times_old=times_old, sim_time=sim_time, v0_intersection=True, g_threshold_gd=g_threshold_gd, g_threshold_dg=g_threshold_dg, d_threshold_gd=d_threshold_gd, d_threshold_dg=d_threshold_dg)
                        
                        # v0 threshold is crossed last
                        else:   
                            v, intersections, time_intersections, v_intersections, vs_old, times_old, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=v, v1=v1, v0_bounds=v0_bounds, v1_bounds=v1_bounds, time=time, intersections=intersections, time_intersections=time_intersections, v_intersections=v_intersections, vs_old=vs_old, times_old=times_old, sim_time=sim_time, v0_intersection=False, g_threshold_gd=g_threshold_gd, g_threshold_dg=g_threshold_dg, d_threshold_gd=d_threshold_gd, d_threshold_dg=d_threshold_dg)
                        
                        # now both thresholds crossed
                        g_threshold_gd = True
                        g_threshold_dg = True
                        d_threshold_gd = False
                        d_threshold_dg = False
                
                # if only one condition holds, store which one (at step i)
                # needed for step i+1, when both thresholds are fulfilled, but need 
                # to know which of these conditions was previously false at i
                if test_threshold_gd>v0_t:
                    g_threshold_gd = True
                else:
                    g_threshold_gd = False
                    
                if test_threshold_dg>v1:
                    g_threshold_dg = True
                    
                else:
                    g_threshold_dg = False
                    
                    
                    
            else :
                # state change
                if test_threshold_dg<v1 and test_threshold_gd<v0_t:
                    # only calculate intersections, if insolation gets smaller v1, but v(t) was already smaller degl. threshold at last timestep
                    if d_threshold_gd and not d_threshold_dg:
                        v, intersections, time_intersections, v_intersections, vs_old, times_old, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=v, v1=v1, v0_bounds=v0_bounds, v1_bounds=v1_bounds, time=time, intersections=intersections, time_intersections=time_intersections, v_intersections=v_intersections, vs_old=vs_old, times_old=times_old, sim_time=sim_time, v0_intersection=False, g_threshold_gd=g_threshold_gd, g_threshold_dg=g_threshold_dg, d_threshold_gd=d_threshold_gd, d_threshold_dg=d_threshold_dg)
                        
                    # only calculate intersections, if v(t) gets smaller than degl. threshold, but insolation was already smaller than v1 at last timestep
                    elif d_threshold_dg and not d_threshold_gd:
                        v, intersections, time_intersections, v_intersections, vs_old, times_old, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=v, v1=v1, v0_bounds=v0_bounds, v1_bounds=v1_bounds, time=time, intersections=intersections, time_intersections=time_intersections, v_intersections=v_intersections, vs_old=vs_old, times_old=times_old, sim_time=sim_time, v0_intersection=True, g_threshold_gd=g_threshold_gd, g_threshold_dg=g_threshold_dg, d_threshold_gd=d_threshold_gd, d_threshold_dg=d_threshold_dg)    
                        
                    # if both thresholds were crossed at same timestep
                    else: 
                        # calculate both intersection points
                        intersect_v0, time_v0 = calc_intersect_v0(i, v, v0_bounds, time)
                        intersect_v1, time_v1 = calc_intersect_v1(i, v1, v1_bounds, time)
                        
                        # choose the later (=smaller) intersection point
                        if time_v0 <= time_v1:   # v0 threshold is crossed last
                            v, intersections, time_intersections, v_intersections, vs_old, times_old, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=v, v1=v1, v0_bounds=v0_bounds, v1_bounds=v1_bounds, time=time, intersections=intersections, time_intersections=time_intersections, v_intersections=v_intersections, vs_old=vs_old, times_old=times_old, sim_time=sim_time, v0_intersection=True, g_threshold_gd=g_threshold_gd, g_threshold_dg=g_threshold_dg, d_threshold_gd=d_threshold_gd, d_threshold_dg=d_threshold_dg)
                        
                        # v0 threshold is crossed last
                        else:   
                            v, intersections, time_intersections, v_intersections, vs_old, times_old, g_threshold_gd, g_threshold_dg, d_threshold_gd, d_threshold_dg = intersection(i=i, v=v, v1=v1, v0_bounds=v0_bounds, v1_bounds=v1_bounds, time=time, intersections=intersections, time_intersections=time_intersections, v_intersections=v_intersections, vs_old=vs_old, times_old=times_old, sim_time=sim_time, v0_intersection=False, g_threshold_gd=g_threshold_gd, g_threshold_dg=g_threshold_dg, d_threshold_gd=d_threshold_gd, d_threshold_dg=d_threshold_dg)

                        
                        # now both thresholds crossed
                        d_threshold_gd = True
                        d_threshold_dg = True
                        g_threshold_gd = False
                        g_threshold_dg = False
                    
                # if only one condition holds, store which one (at step i)
                # needed for step i+1, when both thresholds are fulfilled, but need 
                # to know which of these conditions was previously false at i
                if test_threshold_gd<v0_t:
                    d_threshold_gd = True
                else:
                    d_threshold_gd = False

                if test_threshold_dg<v1:
                    d_threshold_dg = True
                else:                
                    d_threshold_dg = False
                    
            
            if S[0]=="g":
                state.append(0)
            else:
                state.append(1)
                
            k1 = Phi(2*i,v[i])
            k2 = Phi(2*i+1,v[i]+k1*step/2.)
            k3 = Phi(2*i+1,v[i]+k2*step/2.)
            k4 = Phi(2*i+2,v[i]+step*k3)
            v[i+1] = v[i] + step/6.*(k1+2*k2+2*k3+k4)
        return v, time_intersections, v_intersections, vs_old, times_old
    
    # calculates RMSE between icevolume with intersection points and sea data with interpolated intersection points
    def rmse_with_intersections(icevolume, sea, time, time_sea, time_intersections):
        # only use past times + icevolumes, no future
        time_intersections_past = np.array(time_intersections)[np.array(time_intersections)>=0]
        v_intersections_past = v_intersections[:len(time_intersections_past)]
        
        # Step 1: Concatenate time arrays
        time_combined = np.concatenate((time_sea, time_intersections_past))

        # Step 2: Sort the combined time array and get the sorted indices + inverse since runs from 2.6 Myr-0
        sorted_indices = np.argsort(time_combined)
        time_new = time_combined[sorted_indices][::-1]

        # Step 3: Create a combined icevolume array
        icevolume_combined = np.concatenate((icevolume[:len(sea)], v_intersections_past))

        # Step 4: Sort icevolume according to the sorted time array + inverse since runs from 2.6 Myr-0
        icevolume_new = icevolume_combined[sorted_indices][::-1]
        
        # Step 5: Interpolate sea data to include values at inersection times
        sea_interp = np.interp(time_new, time_sea[::-1], sea[::-1])
        
        # Step 6: Calculate RMSE
        rmse_new = root_mean_squared_error(sea_interp, icevolume_new)
        
        return rmse_new
    
# calculate no intersections
else:
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
            t = -1*start_year-(i*sim_time/time_steps)
            
            # thresholds for state changes (use Esi, Eco, EnO at full time steps only)
            test_threshold_gd = kEsi*Esi[2*i]+kEco*Eco[2*i]+kO*EnO[2*i]+v[i]
            test_threshold_dg = kEsi*Esi[2*i]+kEco*Eco[2*i]+kO*EnO[2*i]     
            
            # t>t2: Before Ramp
            if t>t2:
                v0_t = v0_prime
                
            # t2>=t>=t1: Ramp
            elif t2>=t>=t1:
                # v0(t) = v0 - (v0-v0')/(t2-t1) * (t-t1)
                v0_t = v0 - ((v0-v0_prime)/np.abs(t2-t1)) * np.abs(t-t1) 
            
            # t1>t: After Ramp
            else:
                v0_t = v0
            
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
def calc_bounds(time_steps):
    v0s = np.zeros(2*time_steps+1)
    v0_bounds = np.zeros(2*time_steps+1)
    v1_bounds = np.zeros(2*time_steps+1)
    # v0_bound = v0(t)-insolation (eq.5 from Legrain et.al.)
    
    # complete simulation time
    sim_time = abs(start_year)+abs(future_time)
    print('sim_time: ', sim_time)
    print('time_steps: ', time_steps)
    
    for i in range(2*time_steps+1):
        # current time t
        t = -1*start_year-((i*sim_time/time_steps)/2)    
        
        # t>t2: Before Ramp
        if t>t2:
            v0_t = v0_prime
            
        # t2>=t>=t1: Ramp
        elif t2>=t>=t1:
            # v0(t) = v0 - (v0-v0')/(t2-t1) * (t-t1)
            v0_t = v0 - ((v0-v0_prime)/np.abs(t2-t1)) * np.abs(t-t1) 
        
        # t1>t: After Ramp
        else:
            v0_t = v0
        
        v0s[i] = v0_t
        v0_bounds[i] = v0s[i]-kEsi*Esi[i]-kEco*Eco[i]-kO*EnO[i]
        v1_bounds[i] = kEsi*Esi[i]+kEco*Eco[i]+kO*EnO[i]
    
    time_bounds = np.arange(-start_year*1e3, -future_time*1e3-1, -resolution*0.5)*1e-3
    return (time_bounds, v0s, v0_bounds, v1_bounds)



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
aEco = StartPosition[1]
aO = StartPosition[2]
ag = StartPosition[3]
ad = StartPosition[4]
taud0 = StartPosition[5]
kEsi = StartPosition[6]
kEco = StartPosition[7]
kO = StartPosition[8]
v0 = StartPosition[9]
v1 = StartPosition[10]
vi = StartPosition[11]
v0_prime = StartPosition[12]
taud0_prime = StartPosition[13]
t2 = StartPosition[14]
t1 = StartPosition[15]
    
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
Esi = interpol(Esi,2)
Eco = interpol(Eco,2)
EnO = interpol(EnO,2)
time_halfsteps = interpol(time,2)

# calculate bounds for deglaciation/glaciation
time_bounds, v0s, v0_bounds, v1_bounds = calc_bounds(time_steps)

# calculate gradients
# v0_bounds_grad = np.gradient(v0_bounds, resolution*1e-3)

##########################################################
#Modelling of the ice volume for the best parameters fit
if intersections:
    icevolume, time_intersections, v_intersections, vs_old, times_old = modelledVolume(start_year, future_time, vi, time_steps)
else:
    icevolume = modelledVolume(start_year, future_time, vi, time_steps)

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

# index to the left of ramp start (t2)
pre_ramp = int(np.floor(-start_year-t2) / step)
print(f'Time of Ramp start: {time[pre_ramp]}')

# index to the left of ramp start (t1)
post_ramp = int(np.floor(-start_year-t1) / step)
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
if intersections:
    rmse_new = rmse_with_intersections(icevolume, sea, time, time_sea, time_intersections)
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
if intersections:
    print(f"      RMSE (with intersections) = {rmse_new}")
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
if gap!=None:
    ax1.axvspan(xmin=-gap[0], xmax=-gap[1], facecolor=bright.yellow, alpha=0.5, label='Gap')
plt.vlines(x=t2, ymin=np.min([np.min(icevolume-10),v0_prime-10]), ymax=np.max([np.max(icevolume+10),v0+10]), linestyle='-', color=bright.green, label=f'Start of RAMP: t2={int(t2)} kyr')
ax1.plot(time_bounds, v0s, linestyle="--" , color=bright.black, label=r"Deglaciation threshold: $v_0(t)$")
plt.vlines(x=t2, ymin=np.min([np.min(icevolume-10),v0_prime-10]), ymax=np.max([np.max(icevolume+10),v0+10]), linestyle='-', color=bright.green, label=f'Start of RAMP: t2={int(t2)} kyr')
plt.vlines(x=t1, ymin=np.min([np.min(icevolume-10),v0_prime-10]), ymax=np.max([np.max(icevolume+10),v0+10]), linestyle='-', color=bright.green, label=f'End of RAMP: t1={int(t1)} kyr')

ax2 = ax1.twinx()
ax2.plot(time, state, linewidth = 0.8, color=bright.grey, label='Model state')

# plt.xticks([0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000])
# ax2.set_yticks([1,0], ['Interglacial', 'Glacial'], rotation=22.5)
ax2.set_yticks([1,0])
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
# plt.savefig('../Plots/RAMP5.png', dpi=500, bbox_inches='tight')
plt.show()


###############################################################################
# 2ND PLOT: Intersections Plot
if intersections:
    fig, ax1 = plt.subplots(figsize=(20,5))
    
    ax1.plot(time, icevolume, marker='o', color=bright.blue, label="Model")
    ax1.plot(time_bounds, v0_bounds, marker='o', color='orange', label=r"v0 bounds")
    ax1.plot(time_bounds, v1_bounds, marker='o', color='brown', label=r"v1 bounds")
    ax1.scatter(time_intersections, v_intersections, s=100, label='Intersections', color='purple', zorder=10)
    ax1.scatter(times_old, vs_old, s=100, label='old vs', color='red', zorder=10)
    ax1.hlines(v1, xmin=-start_year, xmax=-future_time)
    
    ax1.set_xticks(np.arange(-start_year, -future_time-1, -200))
    ax1.invert_yaxis()
    plt.gca().invert_xaxis()
    ax1.set_xlabel("Age (ka)",weight='bold')
    ax1.set_ylabel("Ice volume (m sl)",weight='bold')
    plt.legend()
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
# # plt.savefig('../Plots/RAMP5_periodogram.png', dpi=300)
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
# # plt.savefig('../Plots/RAMP5_periodogram.png', dpi=300)
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
# plt.savefig('../Plots/RAMP5_periodogram.png', dpi=300)
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
# # plt.savefig('../Plots/RAMP5_spectrogram.png', dpi=300)
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
# plt.savefig(f'../Plots/test2.png', dpi=300)
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


###############################################################################
# 5th Plot: residuals (model-data) for the best fit StartPosition
boundary = 20

plt.figure(figsize=(20,4), dpi=300)
plt.plot(time_sea, residuals, color=bright.green, label="Model")
plt.hlines(0, -start_year, -future_time-1, color='grey')
plt.hlines(boundary, -start_year, 0, linestyle='--', color='grey')
plt.hlines(-boundary, -start_year, 0, linestyle='--', color='grey')

plt.fill_between(time_sea, residuals, boundary, where=(np.array(residuals) >= boundary), color='red', alpha=0.3)
plt.fill_between(time_sea, residuals, -boundary, where=(np.array(residuals) <= -boundary), color='blue', alpha=0.3)

plt.xticks(np.arange(-start_year, -1, -200))
plt.yticks([boundary, 0, -boundary])


plt.xlim(-start_year, 0)
# plt.ylim(-50, 50)
# plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()

plt.xlabel("Age (ka)",weight='bold')
plt.ylabel('Residuals (model-data)(msl)',weight='bold')
plt.show()

###############################################################################
# Save data to binary file
if save_data!=None:
    # store all relevant data for later plotting
    data = {'rmse': rmse, 
            'smape': SMAPE, 
            'R2': R2,
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
            't1': t1, 
            't2': t2, 
            'taud0': taud0, 
            'taud0_prime': taud0_prime,
            'v0_prime': v0_prime, 
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
            'Eco': Eco, 
            'EnO': EnO}
            
    
    if intersections:
        data.update({
            'time_intersections': time_intersections, 
            'v_intersections': v_intersections,
            'times_old': times_old, 
            'vs_old': vs_old
            })    
    
    
    # Write to binary file
    with open(save_data, 'wb') as f:
        pickle.dump(data, f)