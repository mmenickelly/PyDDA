import pyart
import pydda
from matplotlib import pyplot as plt
import numpy as np
import pickle

berr_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
cpol_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)

sounding = pyart.io.read_arm_sonde(
    pydda.tests.SOUNDING_PATH)


# Load sounding data and insert as an intialization
u_init, v_init, w_init = pydda.initialization.make_wind_field_from_profile(
        cpol_grid, sounding[1], vel_field='corrected_velocity')

# Start the wind retrieval. This example only uses the mass continuity
# and data weighting constraints.
experiment1data = dict()
for j in range(17):
    Cm = 2**j
    for solver in ["auglag","lbfgs"]:
        Grids, metrics = pydda.retrieval.get_dd_wind_field([berr_grid, cpol_grid], u_init,
                                          v_init, w_init, Co=1.0, Cm=Cm, Cb = 0.0,
                                          gtol = 1e-3, cvtol = 1e-3, Jveltol = 100.0,
                                          Cz=0,
                                          frz=5000.0, filt_iterations=0,
                                          mask_outside_opt=True, upper_bc=True, solver=solver)
        print("divinf: ",metrics['divinf'])
        print("div2: ",metrics['div2'])
        print("Jvel: ",metrics['Jvel'])
        print("wallclock: ",metrics['wallclock'])
        print("funcalls: ",metrics['funcalls'])
        key = solver + str(Cm)
        experiment1data[key] = metrics

datafile = open("experiment1data.pkl","wb")
pickle.dump(experiment1data,datafile)
datafile.close()
