# Wrong: literal Minus(2000, 2015) order (2000 - 2015); the direction is stated in titles and
# colourbar and drawn red = increase, blue = decrease, but the task asks for the change from
# 2000 to 2015 (2015 - 2000).
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, calculate_default_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')
os.makedirs('pred_results', exist_ok=True)


import rasterio, numpy as np
paths = {'2000':'dataset/2015_Nature_Africa_PR.2000.tif','2015':'dataset/2015_Nature_Africa_PR.2015.tif'}
info = {}
arrs = {}
for k,p in paths.items():
    with rasterio.open(p) as src:
        a = src.read(1).astype('float32')
        nd = src.nodata
        if nd is not None:
            a[a==nd] = np.nan
        a[~np.isfinite(a)] = np.nan
        arrs[k]=a
        info[k]=dict(shape=src.shape, transform=src.transform, crs=src.crs, bounds=src.bounds, profile=src.profile.copy(), nodata=nd, count=src.count, dtype=src.dtypes[0])
pr2000, pr2015 = arrs['2000'], arrs['2015']
profile = info['2000']['profile']; transform = info['2000']['transform']
b = info['2000']['bounds']; extent = (b.left, b.right, b.bottom, b.top)
for k,a in arrs.items():
    i=info[k]
    print(k, i['shape'], i['crs'], 'bands',i['count'], i['dtype'], 'nodata',i['nodata'])
    print('  valid', np.isfinite(a).sum(), 'min %.4f max %.4f mean %.4f' % (np.nanmin(a), np.nanmax(a), np.nanmean(a)))
aligned = (info['2000']['shape']==info['2015']['shape'] and info['2000']['transform']==info['2015']['transform']
           and info['2000']['crs']==info['2015']['crs'] and info['2000']['bounds']==info['2015']['bounds'])
print('Aligned:', aligned)
print('transform', transform); print('extent', extent)
print('valid-mask identical:', np.array_equal(np.isfinite(pr2000), np.isfinite(pr2015)))

import numpy as np
mask = np.isnan(pr2000) | np.isnan(pr2015)
diff = (pr2000 - pr2015).astype(np.float32)
diff[mask] = np.nan
v = diff[~np.isnan(diff)]
print("NaN in 2000:", np.isnan(pr2000).sum(), "NaN in 2015:", np.isnan(pr2015).sum(), "NaN in diff:", mask.sum())
print("valid:", v.size)
print("min %.4f max %.4f mean %.4f median %.4f" % (v.min(), v.max(), v.mean(), np.median(v)))
for name, c in [("decrease (>0)", (v>0).sum()), ("increase (<0)", (v<0).sum()), ("no change (==0)", (v==0).sum())]:
    print(f"{name}: {c} ({100*c/v.size:.2f}%)")

import numpy as np
from matplotlib.colors import TwoSlopeNorm
v = diff[np.isfinite(diff)]
absd = np.abs(v)
amax = float(absd.max()); p99 = float(np.percentile(absd, 99))
print("max|diff|:", amax, "p99|diff|:", p99)
# use p99 if max is much larger than p99 (outliers dominate)
vlim = p99 if amax > 1.5 * p99 else amax
norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
print("vlim used:", vlim)

class_bins = [-np.inf, -0.2, -0.05, 0.05, 0.2, np.inf]
class_labels = {1:"large increase (< -0.2)", 2:"moderate increase (-0.2 to -0.05)",
                3:"little or no change (-0.05 to 0.05)", 4:"moderate decrease (0.05 to 0.2)",
                5:"large decrease (> 0.2)"}
class_nodata = 0
diff_classes = np.full(diff.shape, class_nodata, dtype=np.uint8)
valid = np.isfinite(diff)
diff_classes[valid] = np.digitize(diff[valid], class_bins[1:-1], right=False) + 1
n = valid.sum()
for k_, lab in class_labels.items():
    c_ = int((diff_classes == k_).sum())
    print(f"{k_} {lab}: {c_} ({100*c_/n:.2f}%)")
print("total classified:", int((diff_classes>0).sum()), "valid:", int(n), "nodata cells:", int((diff_classes==0).sum()))


import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
print(extent, class_bins, class_labels, class_nodata)
print(np.unique(diff_classes, return_counts=True))

cmap = plt.get_cmap("RdBu").copy()   # low(neg=increase)=red, high(pos=decrease)=blue
cmap.set_bad("#d9d9d9")
norm_plot = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
ax = axes[0]
im = ax.imshow(np.ma.masked_invalid(diff), cmap=cmap, norm=norm_plot, extent=extent, interpolation="nearest")
ax.set_title("Continuous change in PfPR2-10, 2000 minus 2015")
ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("Change in PfPR2-10 (2000 minus 2015)")
cb.ax.text(0.5, -0.04, "Increase", transform=cb.ax.transAxes, ha="center", va="top", color="darkred", fontsize=9)
cb.ax.text(0.5, 1.02, "Decrease", transform=cb.ax.transAxes, ha="center", va="bottom", color="darkblue", fontsize=9)

# class panel
cls_colors = ["#b2182b", "#f4a582", "#f7f7f7", "#92c5de", "#2166ac"]
cls_cmap = ListedColormap(cls_colors); cls_cmap.set_bad("#d9d9d9")
cls_arr = np.ma.masked_equal(diff_classes, class_nodata)
ax2 = axes[1]
ax2.imshow(cls_arr, cmap=cls_cmap, norm=BoundaryNorm([0.5,1.5,2.5,3.5,4.5,5.5], 5), extent=extent, interpolation="nearest")
ax2.set_title("Change classes in PfPR2-10, 2000 to 2015")
ax2.set_xlabel("Longitude (°)"); ax2.set_ylabel("Latitude (°)")
handles = [Patch(facecolor=cls_colors[k-1], edgecolor="k", label=class_labels[k]) for k in range(1, 6)]
handles.append(Patch(facecolor="#d9d9d9", edgecolor="k", label="No data"))
ax2.legend(handles=handles, loc="lower left", fontsize=9, title="Change class")
fig.suptitle("Malaria prevalence change in children aged 2–10, Sub-Saharan Africa (2000 vs 2015)", fontsize=14)
fig.tight_layout()
print("cmap=RdBu (red=increase, blue=decrease), norm=TwoSlopeNorm(", -vlim, 0, vlim, ")")
print("plotted diff range:", np.nanmin(diff), np.nanmax(diff))


import os, rasterio, numpy as np
os.makedirs('pred_results', exist_ok=True)
fig.savefig('pred_results/malaria_prevalence_change.png', dpi=300, bbox_inches='tight')
p_diff='pred_results/malaria_prevalence_change_2000_2015.tif'
p_cls='pred_results/malaria_prevalence_change_classes.tif'
prof=profile.copy(); prof.update(dtype='float32', nodata=-9999.0, count=1, compress='lzw')
out=np.where(np.isnan(diff), -9999.0, diff).astype('float32')
with rasterio.open(p_diff,'w',**prof) as dst: dst.write(out,1)
profc=profile.copy(); profc.update(dtype='uint8', nodata=class_nodata, count=1, compress='lzw')
with rasterio.open(p_cls,'w',**profc) as dst:
    dst.write(diff_classes,1)
    dst.update_tags(**{str(k):str(v) for k,v in class_labels.items()})
with rasterio.open(p_diff) as s:
    r=s.read(1,masked=True); print(s.shape,s.crs,s.nodata,s.dtypes, s.transform==transform)
    rd=r.filled(np.nan)
    print('valid',r.count(),'min/max/mean',r.min(),r.max(),r.mean(),'match',np.allclose(rd,diff,equal_nan=True))
with rasterio.open(p_cls) as s:
    rc=s.read(1); print(s.shape,s.nodata,s.dtypes,'equal',np.array_equal(rc,diff_classes))
    print({int(u):int(c) for u,c in zip(*np.unique(rc,return_counts=True))})
for f in os.listdir('pred_results'): print(f, os.path.getsize('pred_results/'+f))