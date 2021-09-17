import pydda
import pyart
import numpy as np

Grid_list = [pyart.io.read_grid('Grid0_regular_cpol.nc'),pyart.io.read_grid('Grid1_regular_cpol.nc')]
x = np.load('w_wrf.npz')
z_prof = np.arange(0,20000,125)
profile = pyart.core.HorizontalWindProfile.from_u_and_v(z_prof,x['u_vert'], x['v_vert'])
u_init, v_init, w_init = pydda.initialization.make_wind_field_from_profile(Grid_list[0], profile, vel_field="VT")
Grids = pydda.retrieval.get_dd_wind_field(Grid_list, u_init, v_init, w_init, vel_name="VT", refl_field="DT", gtol=1e-3, cvtol=1e-3, Cm = 1500.0, Jveltol = 100.0)
