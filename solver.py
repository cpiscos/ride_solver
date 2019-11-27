from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint, Bounds, minimize


def get_simulated_accel_and_velocity(x, slope, time, g=9.81):
    gravity = g * np.sin(slope)
    velocity = [0]
    sim_accels = []
    td = time[1:] - time[:-1]
    for i, input_accel in enumerate(x):
        if i != 0:
            velocity.append(new_velocity(sim_accels[-1], velocity[-1], td[i]))
        sim_accel = gravity[i] + input_accel - 0.02 * velocity[-1] ** 2
        sim_accels.append(sim_accel)
    velocity.append(new_velocity(sim_accels[-1], velocity[-1], td[-1]))
    return np.array(sim_accels), np.array(velocity)


def new_velocity(sim_accel, prev_velocity, td):
    return sim_accel * td + prev_velocity


def get_power(x, mass=150):
    power = x * mass * get_simulated_accel_and_velocity(x)[1][:-1]
    return power


def get_centripetal_accel(x, radii):
    return get_simulated_accel_and_velocity(x)[1][:-1] ** 2 / radii


def get_mags(x):
    centripetal_accel = get_centripetal_accel(x)
    return np.sqrt(centripetal_accel ** 2 + x ** 2)


def objective(x, time, with_velocity=True):
    velocity = get_simulated_accel_and_velocity(x)[1]
    distance = np.mean(velocity) * time[-1]
    if with_velocity:
        return -distance + -np.mean(velocity)
    else:
        return -distance


if __name__ == '__main__':
    df = pd.read_excel("optimize 1.xlsx", header=2)
    test_vars = df["rider input accel"].values[:-1]
    constant = df.loc[:, ["time", "slope", "safe max accel", "radius", "safe min accel"]]
    variables = len(constant) - 1

    get_simulated_accel_and_velocity = partial(get_simulated_accel_and_velocity, slope=constant["slope"][:-1].values,
                                               time=constant["time"].values)
    get_centripetal_accel = partial(get_centripetal_accel, radii=constant["radius"][1:].values)
    objective = partial(objective, time=constant["time"].values, with_velocity=False)

    safe_min_accel_constraint = Bounds(constant["safe min accel"][1:], np.inf)
    power_constraint = NonlinearConstraint(get_power, -np.inf, 500)
    mags_constraint = NonlinearConstraint(get_mags, -np.inf, constant["safe max accel"][1:])
    result = minimize(objective, np.random.randn(variables), bounds=safe_min_accel_constraint,
                      constraints=[power_constraint, mags_constraint],
                      method='trust-constr')
    print("Distance using excel inputs: {}".format(-objective(test_vars, with_velocity=False)))
    print("Distance from scipy inputs: {}".format(-objective(result.x, with_velocity=False)))
    print(("{:.16f}\n"*len(result.x)).format(*result.x.tolist()))
