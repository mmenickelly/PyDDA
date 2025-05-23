import pyart
import pydda
from matplotlib import pyplot as plt
import numpy as np

berr_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
cpol_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)

sounding = pyart.io.read_arm_sonde(
    pydda.tests.SOUNDING_PATH)


# Load sounding data and insert as an intialization
u_init, v_init, w_init = pydda.initialization.make_wind_field_from_profile(
        cpol_grid, sounding[1], vel_field='corrected_velocity')

# Start the wind retrieval. This example only uses the mass continuity
# and data weighting constraints.
Grids, metrics = pydda.retrieval.get_dd_wind_field([berr_grid, cpol_grid], u_init,
                                          v_init, w_init, Co=1.0, Cm=1500.0, Cb = 0.0,
                                          gtol = 1e-3, cvtol = 1e-3, Jveltol = 100.0,
                                          Cz=0,
                                          frz=5000.0, filt_iterations=0,
                                          mask_outside_opt=True, upper_bc=True, solver="auglag")
print("divinf: ",metrics['divinf'])
print("div2: ",metrics['div2'])
print("Jvel: ",metrics['Jvel'])
print("wallclock: ",metrics['wallclock'])
print("funcalls: ",metrics['funcalls'])

# Plot a horizontal cross section
plt.figure(figsize=(9, 9))
pydda.vis.plot_horiz_xsection_barbs(Grids, background_field='reflectivity',
                                    level=6,
                                    w_vel_contours=[3, 6, 9, 12, 15],
                                    barb_spacing_x_km=5.0,
                                    barb_spacing_y_km=15.0)

plt.savefig('horiz_cross_sect.png',format='png')

# Plot a vertical X-Z cross section
plt.figure(figsize=(9, 9))
pydda.vis.plot_xz_xsection_barbs(Grids, background_field='reflectivity',
                                 level=40,
                                 w_vel_contours=[3, 6, 9, 12, 15],
                                 barb_spacing_x_km=10.0,
                                 barb_spacing_z_km=2.0)
plt.savefig('vert_xz_cross_sect.png',format='png')

# Plot a vertical Y-Z cross section
plt.figure(figsize=(9, 9))
pydda.vis.plot_yz_xsection_barbs(Grids, background_field='reflectivity',
                                 level=40,
                                 w_vel_contours=[3, 6, 9, 12, 15],
                                 barb_spacing_y_km=10.0,
                                 barb_spacing_z_km=2.0)
plt.savefig('vert_yz_cross_sect.png',format='png')
