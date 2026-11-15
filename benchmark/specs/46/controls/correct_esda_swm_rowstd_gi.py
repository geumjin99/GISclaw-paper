# Correct: esda G_Local (Gi, star=False, row-standardised) with the supplied network SWM; rate per km-year; FDR bins
import geopandas as gpd, numpy as np, libpysal, matplotlib.pyplot as plt
from esda.getisord import G_Local
from esda import fdr

def main():
    roads = gpd.read_file("dataset/roads.shp")
    cr = gpd.read_file("dataset/crashes.shp")
    wk = cr[~cr.CRASH_DOW.isin(["Saturday", "Sunday"]) & (cr.CRASH_HOD >= 15) & (cr.CRASH_HOD <= 17)]
    nn = gpd.sjoin_nearest(wk[["geometry"]], roads[["UniqID", "geometry"]], max_distance=0.25 * 5280, distance_col="d")
    nn = nn.loc[~nn.index.duplicated()]
    per_seg = nn.groupby("UniqID").size().reindex(roads.UniqID, fill_value=0).to_numpy(float)
    rate_km = per_seg / (6 * roads.Shape_Leng.to_numpy() * 0.0003048)
    w = libpysal.io.open("dataset/nwswm360ft.swm", "r").read()
    w.transform = "R"
    g = G_Local(rate_km, w, transform="R", star=False, permutations=0)
    z = g.Zs
    p = g.p_norm
    cut = fdr(p, 0.05)
    roads.assign(z=z, hot=(z > 0) & (p <= cut)).plot(column="z", cmap="coolwarm", vmin=-4, vmax=4, linewidth=0.5,
                                                     figsize=(8, 10))
    plt.savefig("pred_results/peak_crash_hotspots.png")

main()
