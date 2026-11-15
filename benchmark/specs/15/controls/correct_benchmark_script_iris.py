# Correct: Iris solution; Iris loads the rotated-pole coordinate system from the file.
# Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
import matplotlib.pyplot as plt
import numpy.ma as ma
import iris
import iris.quickplot as qplt
cube = iris.load_cube("dataset/space_weather.nc", "TEC")
cube.data = ma.masked_less(cube.data, 0)
qplt.contourf(cube, 100)
plt.title("Total Electron Content")
plt.gca().stock_img()
plt.gca().coastlines()
plt.savefig("pred_results/TEC_vis.png")
