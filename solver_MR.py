from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint, Bounds, minimize
from time import time
import matplotlib.pyplot as plt


def get_simulated_accel_and_velocity(x, slope, time, g=9.81):
    
    gravity = g * np.sin(slope)
    velocity = [0]
    sim_accels = []
    td = time[1:] - time[:-1]
    
    for i, input_accel in enumerate(x):
        if i > 0:
            velocity.append(new_velocity(sim_accels[-1], velocity[-1], td[i-1]))
        sim_accel = gravity[i] + input_accel - 0.02 * velocity[-1] ** 2
        sim_accels.append(sim_accel)
        
    return np.array(sim_accels), np.array(velocity)


def new_velocity(sim_accel, prev_velocity, td):
    return sim_accel * td + prev_velocity


def get_power(x, mass=150):
    power = x * mass * get_simulated_accel_and_velocity(x)[1]
    return power


def get_centripetal_accel(x, radii):
    return get_simulated_accel_and_velocity(x)[1] ** 2 / radii


def get_mags(x):
    centripetal_accel = get_centripetal_accel(x)
    return np.sqrt(centripetal_accel ** 2 + x ** 2)


def objective(x, time):
    
    velocity = get_simulated_accel_and_velocity(x)[1]
    distance = np.mean(velocity) * time[-1]
    # if with_velocity:
    #     return -distance + -np.mean(velocity)
    # else:
    
    return -distance


print_vars_for_excel = True
t = np.arange(0,50)
start_time = time()
# df = pd.read_excel("optimize 60.xlsx", header=2)
# test_vars = df["rider input accel"].values
angle = np.sin(t)*0
max_acc = 0.1*np.sin(t)+2
radius = (np.sin(t*0.1)**2)*100+3
min_acc = -max_acc
# constant = df.loc[:, ["time", "angle", "safe max accel", "radius", "safe min accel"]]
guess = [0]*len(t)

get_simulated_accel_and_velocity = partial(get_simulated_accel_and_velocity,
                                           slope=angle,
                                           time=t)

get_centripetal_accel = partial(get_centripetal_accel, radii=radius)

objective = partial(objective, time=t)

safe_min_accel_constraint = Bounds(min_acc, np.inf)

power_constraint = NonlinearConstraint(get_power, -np.inf, 500)

mags_constraint = NonlinearConstraint(get_mags, -np.inf, max_acc)

opt_start_time = time()

result = minimize(objective, guess, bounds=safe_min_accel_constraint,
                  constraints=[power_constraint, mags_constraint],
                  method='SLSQP')

inputs = result['x']

sim_acc,vel = get_simulated_accel_and_velocity(x=inputs,slope=angle,time=t)
cen_acc = get_centripetal_accel(x=inputs,radii=radius)
mag_acc = get_mags(inputs)
power = get_power(inputs)

fig,[ax,ax2] = plt.subplots(1,2,figsize=(20,10))
ax.plot(t,max_acc,lw=5,c='g',label='max_acc')
ax.plot(t,min_acc,lw=5,c='r',label='min_acc')
ax.plot(t,inputs,'red',label='rider optimum')
ax.fill_between(t,0,inputs,where=inputs <0,facecolor='red',alpha=0.2,interpolate=True)
ax.fill_between(t,0,inputs,where=inputs >= 0,facecolor='green',alpha=0.2,interpolate=True)
ax.plot(t,cen_acc,'gold',label='centripital')
ax.plot(t,mag_acc,'b--',lw=5,label='magnitude')


ax2.plot(t,vel,'r',label = 'velocity')
ax3 = ax2.twinx()
ax3.plot(t,power,'g--',lw=5,label = 'power')

ax.legend(fontsize='x-large')
ax2.legend(fontsize='x-large')
ax3.legend(fontsize='x-large')
plt.show()
 



print("Total time taken: {:.4}s".format(time()-start_time))
print("Optimization time taken: {:.4}s".format(time()-opt_start_time))
# print("Distance using excel inputs: {}".format(-objective(test_vars)))
print("Distance using scipy inputs: {}".format(-objective(result.x)))
# if print_vars_for_excel:
    # print(("Vars:\n" + "{:.10f}\n"*variables).format(*result.x.tolist()))
