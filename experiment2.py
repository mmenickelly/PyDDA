import pyart
import pydda
from matplotlib import pyplot as plt
import numpy as np
import pickle

Grid_list = [pyart.io.read_grid('Grid0_regular_cpol.nc'),pyart.io.read_grid('Grid1_regular_cpol.nc')]
x = np.load('w_wrf.npz')
z_prof = np.arange(0,20000,125)
profile = pyart.core.HorizontalWindProfile.from_u_and_v(z_prof,x['u_vert'], x['v_vert'])
u_init, v_init, w_init = pydda.initialization.make_wind_field_from_profile(Grid_list[0], profile, vel_field="VT")

# Start the wind retrieval. This example only uses the mass continuity
# and data weighting constraints.
experiment2data = dict()
for j in range(17):
    Cm = 2**j
    for solver in ["auglag"]: #["auglag","lbfgs"]:
        Grids, metrics = pydda.retrieval.get_dd_wind_field(Grid_list, u_init,
                                          v_init, w_init, vel_name="VT", refl_field="DT", Co=1.0, Cm=Cm, Cb = 0.0,
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
        experiment2data[key] = metrics

datafile = open("experiment2data.pkl","wb")
pickle.dump(experiment2data,datafile)
datafile.close()
