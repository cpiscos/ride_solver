import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import time
from functools import partial
from scipy.optimize import NonlinearConstraint, minimize
token = 'pk.eyJ1IjoibWFja2luYXRvciIsImEiOiJjazJtbHY3ZW4wamM4M2NxZXJvYnJjeDhsIn0.cF5Gsxx6_anrbfMUZg-scA'

# DATA
df = pd.read_csv('lap_df.csv').iloc[0:9000]
turn_slices = pd.read_csv('turn slices')
df['Slope'] = np.arctan(df.GPS_Elevation.shift(int(-2000/2)).rolling(2000).mean().diff() / df.Distance.diff())    
df.Slope.fillna(0,inplace=True)
# df.Slope = np.random.uniform(-0.2,0,df.Slope.size)
df.Slope = -0.1

# DESGIN CONSTANTS
g = 9.81
m = 69
CA = 0.357
rho = 1.23
R_c = 0.175
R_r = 0.1
R_w = 0.36
u_r = 0.02
u_s = 0.56
e = 0.01 # embankment angle
STEP = 10
dt = df.Time.diff().mean() * STEP

# =============================================================================
# Constraint functions
def get_power_constraint(x, mass=m):
    power = x * mass * get_simulated_accel_and_velocity_p(x)[1]
    return power


def get_mags_constraint(x):
    # accels = get_simulated_accel_and_velocity_p(x)[0]
    centripetal_accel = get_centripetal_accel_p(x)
    return get_safe_max_accel_p(x) - np.sqrt(centripetal_accel ** 2 + x ** 2)


def get_energy_constraint(x):
    power = get_power_constraint(x).clip(0)
    return sum(power * dt)


def get_safe_min_accel_constraint(x,slope):
    sim_acc,vel,dist,sim_locations = get_simulated_accel_and_velocity_p(x)
    slopes = slope[sim_locations]
    return -np.cos(abs(slopes)) * g * u_s - sim_acc

def get_safe_max_accel_constraint(x,slope):
    sim_acc,vel,dist,sim_locations = get_simulated_accel_and_velocity_p(x)
    slopes = slope[sim_locations]
    return np.cos(abs(slopes)) * g * u_s - sim_acc

def get_speeds_constraint(x):
    sim_acc,vel,dist,sim_locations = get_simulated_accel_and_velocity_p(x)
    return vel

# =============================================================================

# Helper functions
def get_centripetal_accel(x, radius):
    _,speed,_,sim_index = get_simulated_accel_and_velocity_p(x)
    return  speed**2 / radius[sim_index]


def get_mags(x):
    centripetal_accel = get_centripetal_accel_p(x)
    return np.sqrt(centripetal_accel ** 2 + x ** 2)


def get_safe_max_accel(x,slope):
    sim_index = get_simulated_accel_and_velocity_p(x)[-1]
    slopes = slope[sim_index]
    return np.cos(abs(slopes)) * g * u_s

def get_safe_min_accel(x,slope):
    sim_index = get_simulated_accel_and_velocity_p(x)[-1]
    slopes = slope[sim_index]
    return -np.cos(abs(slopes)) * g * u_s
# =============================================================================


def objective(x):
    sim_acc,vel,dist,sim_locations = get_simulated_accel_and_velocity_p(x)
    return -(dist[-1] - dist[0])**2


def get_simulated_accel_and_velocity(x0,initial_velocity, initial_index, slope ,distance):
    sim_velo = [initial_velocity]
    sim_dist = [distance[0]]
    sim_accels = []
    sim_index = []
    for i, x in enumerate(x0):
        
        # Locate simulated position 
        j = np.abs(distance - sim_dist[-1]).argmin()
        o = slope[j]
        
        F_y = m*g*np.cos(o)
        F_drag = 1/2*CA*rho* sim_velo[-1]**2
        
        sim_accel = (-m*g*np.sin(o) + x*m - (F_y*u_r + F_drag))/(m+2)
        new_position = 0.5*sim_accel*dt**2 + sim_velo[-1]*dt + sim_dist[-1]
        
        if i > 0:
            sim_velo.append(sim_accels[-1] * dt + sim_velo[-1])
        sim_dist.append(new_position)
        sim_accels.append(sim_accel)
        sim_index.append(j)
    
    sim_dist.pop(0)
    return np.array(sim_accels), np.array(sim_velo), sim_dist, sim_index


opt_input = []
opt_accels = []
opt_distances = []
opt_locations = []
opt_speeds = []
opt_centripital_accels = []
opt_power = []
opt_magnitude = []
avgtime = []


for s in turn_slices.iloc[0:30,0]:
    
    s = eval(s)
    START = s.start - 10
    STOP = s.stop
    V0 = df.GPS_Speed[START]/3.6
    batch_df = df.iloc[START: STOP :STEP]
    # x0 = np.random.uniform(-3,3,batch_df.shape[0]) 
    x0 = 3*np.cos(np.linspace(1.1,6.4,batch_df.shape[0]))
    print('Optmizing {}, # {} variables'.format(s,batch_df.shape[0]))
    
    # Set up partial function 
    get_simulated_accel_and_velocity_p = partial(get_simulated_accel_and_velocity,
                                                 initial_velocity = V0,
                                                 initial_index = START,
                                                 slope=df["Slope"].iloc[START:STOP].values,
                                                 distance=df["Distance"].iloc[START:STOP].values)
    
    get_centripetal_accel_p = partial(get_centripetal_accel,
                                      radius=df['GPS_Radius'].iloc[START:STOP].values)
    
    get_safe_max_accel_p = partial(get_safe_max_accel,
                                      slope=df['Slope'].iloc[START:STOP].values)
    
    get_safe_min_accel_p = partial(get_safe_min_accel,
                                      slope=df['Slope'].iloc[START:STOP].values)
    
    get_safe_min_accel_constraint_p = partial(get_safe_min_accel,
                                      slope=df['Slope'].iloc[START:STOP].values)
    
    get_safe_max_accel_constraint_p = partial(get_safe_max_accel,
                                      slope=df['Slope'].iloc[START:STOP].values)
    
    # CONSTRAINTS 
    power_constraint = NonlinearConstraint(get_power_constraint,-5000, 10)
    min_accel_constraint = NonlinearConstraint(get_safe_min_accel_constraint_p,-10, 0)
    max_accel_constraint = NonlinearConstraint(get_safe_max_accel_constraint_p,0, 10)
    mags_constraint = NonlinearConstraint(get_mags_constraint,0,10,)
    # energy_constraint = NonlinearConstraint(get_energy_constraint,0, 250)
    # speed_constraint = NonlinearConstraint(get_speeds_constraint,1,20)
    
    start = time.time()
    # Optimize rider input accelerations from START to STOP
    result = minimize(objective,
                      x0,
                      constraints=[
                                    power_constraint,
                                    mags_constraint,
                                    max_accel_constraint,
                                    min_accel_constraint,
                                    # energy_constraint,
                                    # speed_constraint
                                   ],
                      method='SLSQP',
                      options={'maxiter':3})
    
    run = time.time() - start
    avgtime.append(run)
    print('optimized time {}'.format(run))
    print('average {}'.format(np.mean(avgtime)))
   
    # Calculate rider trajectory with optimized inputs
    sim_acc,vel,dist,sim_locations = get_simulated_accel_and_velocity(
                                                            x0=result['x'],
                                                            slope=df["Slope"].values,
                                                            distance=df["Distance"].values,
                                                            initial_velocity = V0,
                                                            initial_index=START)
    # If there is an error in the optmization, skip it.
    if any(np.isnan(result.x)) or any(i < 0 for i in vel):
        print('Skipping {}'.format(s))
        print('Result.x: {}'.format(result.x))
        print('velocity: {}'.format(vel))
        print('index: {}'.format(sim_locations))
        continue
    
    # Store optimized varaibles
    opt_input.extend(result['x'])
    opt_accels.extend(sim_acc)
    opt_locations.extend([START + i for i in sim_locations])
    opt_distances.extend(dist)
    opt_speeds.extend(vel)
    opt_centripital_accels.extend(get_centripetal_accel(result.x,radius=df['GPS_Radius'].iloc[START:STOP].values))
    opt_magnitude.extend(get_mags(result.x))
    opt_power.extend(get_power_constraint(result.x))
    
    plt.plot(sim_locations,result['x'])
                     
    
df['Optimum_Power'] = (-pd.Series(opt_power,index=opt_locations)).clip(0)
df['Optimum_Power'].interpolate(limit=15,limit_area='inside',inplace=True)
df['Optimum_Power'].fillna(0,inplace=True)
df['Optimum_Speed'] = pd.Series(opt_speeds,index=opt_locations).fillna('GPS_Speed')*3.6
df['Optimum_Speed'].interpolate(limit=15,limit_area='inside',inplace=True)
df['Optimum_Speed'].fillna(df['GPS_Speed'],inplace=True)


# =============================================================================
# # Plotting
# =============================================================================
times = df.Time[opt_locations]  
max_speed = (g*u_s * df.GPS_Radius.iloc[opt_locations]) **(1/2)
x = result.x

fig,[ax1,ax2] = plt.subplots(2,1)
ax1.plot(times, np.cos(abs(df.Slope.iloc[opt_locations])) * g * u_s,lw=5,c='g',label='max_acc')
ax1.plot(times, -np.cos(abs(df.Slope.iloc[opt_locations])) * g * u_s,lw=5,c='r',label='min_acc')
ax1.plot(times,opt_input,'k',label='rider optimum')
ax1.fill_between(times,0,opt_input,where=np.array(opt_input) <0,facecolor='red',alpha=0.2,interpolate=True)
ax1.fill_between(times,0,opt_input,where=np.array(opt_input) >= 0,facecolor='green',alpha=0.2,interpolate=True)
ax1.plot(times,opt_centripital_accels,'gold',label='centripital')
ax1.plot(times,opt_magnitude,'k--',lw=5,label='magnitude')
ax1.set_ylabel('acceleration')
ax1.legend(fontsize='x-large',loc='lower right')

ax2.plot(times,opt_speeds,'r',label = 'velocity')
ax2.plot(times,max_speed,'k',label = 'Max speed')
ax2.plot(times,df.GPS_Radius.iloc[opt_locations],'y',label = 'radius')
ax3 = ax2.twinx()
ax3.plot(times,opt_power,'g--',label = 'opt_power')

ax2.set_ylabel('speed')
ax3.set_ylabel('opt_power')
ax2.legend(fontsize='x-large',loc='lower right')
ax2.legend(fontsize='x-large',loc='lower right')
ax3.legend(fontsize='x-large',loc='lower left')
plt.show()


fig = go.Figure()
fig.add_scattermapbox(lat=df[:opt_locations[-1]]["GPS_Latitude"],
                        lon=df[:opt_locations[-1]]["GPS_Longitude"],
                        marker=dict(color=df[:opt_locations[-1]]['Optimum_Power'],
                                      colorscale = 'plasma',size=10),
                        name='Optimum Power',
                        text=df[:opt_locations[-1]][['Avg Power','Optimum_Power']],
                        hovertemplate =
                                        "<b>Brake Power (W)</b><br>" +
                                        "Actual: %{text[0]:1f}<br>"+
                                        "Optimum: %{text[1]:1f}<br><extra></extra>",
                        )

fig.add_scattermapbox(lat=df[:opt_locations[-1]]["GPS_Latitude"],
                        lon=df[:opt_locations[-1]]["GPS_Longitude"],
                        marker=dict(color=df[:opt_locations[-1]]['Avg Power'],
                                    colorscale = 'plasma',size=10),
                                    # size=15,
                                    # opacity=0.9),
                        name='Actual Power',
                        visible='legendonly',
                        text=df[:opt_locations[-1]][['Avg Power','Optimum_Power']],
                        hovertemplate =
                                        "<b>Brake Power (W)</b><br>" +
                                        "Actual: %{text[0]:1f}<br>"+
                                        "Optimum: %{text[1]:1f}<br><extra></extra>",)

fig.update_layout(
                  mapbox=dict(
                        accesstoken=token,
                      zoom=17,
                      style='outdoors',
                      center = go.layout.mapbox.Center(
                                      lat=df.GPS_Latitude.mean(),
                                      lon=df.GPS_Longitude.mean(),
                                      ),
                        ),
                  legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01)
                  )
fig.show(renderer='browser')


