# Correct: the context layers reach the plot through a conditional expression and a helper function.
# One step is wrapped in try/except: it raises after defining the buffer and the error is tolerated.

import geopandas as gpd, os
os.makedirs('pred_results',exist_ok=True)
roads=gpd.read_file('dataset/roads.geojson')
defo=gpd.read_file('dataset/deforestedArea.geojson')
prot=gpd.read_file('dataset/protectedForest.geojson')
plan=gpd.read_file('dataset/planned_road.geojson')
for n,g in [('roads',roads),('defo',defo),('prot',prot),('plan',plan)]:
    print(n,g.crs,len(g),g.total_bounds)
print(roads.Status.value_counts())

crs=plan.crs
R=roads.to_crs(crs);D=defo.to_crs(crs);P=prot.to_crs(crs)
g=D.geometry.iloc[0]; print(g.geom_type, len(g.geoms) if hasattr(g,'geoms') else 1, g.area/1e6)
print(D.total_bounds, R.total_bounds)

try:
    import shapely, numpy as np, matplotlib.pyplot as plt
    from shapely.ops import unary_union
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    buf=plan.buffer(5500)
    bufg=unary_union(buf)
    # clip deforested to buffer region first for speed
    Dc=gpd.clip(D,bufg.buffer(100))
    risk=bufg.difference(unary_union(Dc.geometry).buffer(0))
    riskP=risk.intersection(unary_union(P.geometry).buffer(0))
    print('buffer km2',bufg.area/1e6,'deforested in buffer',(bufg.area-risk.area)/1e6,'risk',risk.area/1e6,'risk in protected',riskP.area/1e6)
    # approximate state boundary
    pts=[]
    for geom in list(D.geometry.iloc[0].geoms)+list(P.geometry)+list(R.geometry.sample(5000,random_state=0)):
        c=shapely.get_coordinates(geom); pts.append(c[::max(1,len(c)//50)])
    pts=np.vstack(pts)
    boundary=shapely.concave_hull(shapely.MultiPoint(pts),ratio=0.05)
    print(boundary.area/1e6)
except Exception as _e:
    print('block failed:', repr(_e)[:200])

dparts=[shapely.make_valid(p) for p in D.geometry.iloc[0].geoms]
dg=gpd.GeoSeries(dparts,crs=crs)
dsel=dg[dg.intersects(bufg.buffer(100))]
dloc=unary_union([shapely.make_valid(x.intersection(bufg)) for x in dsel])
risk=bufg.difference(dloc)
pg=unary_union([shapely.make_valid(x) for x in P.geometry])
riskP=risk.intersection(pg)
print('buffer km2',bufg.area/1e6,'deforested in buffer',dloc.area/1e6,'risk',risk.area/1e6,'risk in protected',riskP.area/1e6)
pts=[]
for geom in dparts+list(P.geometry)+list(R.geometry):
    c=shapely.get_coordinates(geom); pts.append(c[::max(1,len(c)//20)])
pts=np.vstack(pts)
boundary=shapely.concave_hull(shapely.MultiPoint(pts),ratio=0.03)
print(boundary.area/1e6, len(pts))

fig,axes=plt.subplots(1,2,figsize=(18,9))
xmin,ymin,xmax,ymax=bufg.bounds; pad=25000
for i,ax in enumerate(axes):
    gpd.GeoSeries([boundary],crs=crs).boundary.plot(ax=ax,color='black',lw=1.5)
    P.plot(ax=ax,color='#2e8b57',alpha=0.55,edgecolor='darkgreen',lw=0.3)
    D.plot(ax=ax,color='#d2a679',edgecolor='none')
    Rv = R if i==0 else R.cx[xmin-pad:xmax+pad, ymin-pad:ymax+pad]
    Rv.plot(ax=ax,color='grey',lw=0.3 if i==0 else 0.6)
    gpd.GeoSeries([risk],crs=crs).plot(ax=ax,color='red',alpha=0.7,edgecolor='darkred',lw=0.4)
    gpd.GeoSeries([riskP],crs=crs).plot(ax=ax,facecolor='none',hatch='///',edgecolor='yellow',lw=0)
    gpd.GeoSeries([bufg],crs=crs).boundary.plot(ax=ax,color='darkred',lw=1,ls='--')
    plan.plot(ax=ax,color='blue',lw=2)
    ax.set_xticks([]);ax.set_yticks([])
axes[0].set_title('Rondônia: planned road and deforestation risk (overview)')
axes[1].set_xlim(xmin-pad,xmax+pad);axes[1].set_ylim(ymin-pad,ymax+pad)
axes[1].set_title('Zoom: 5.5 km buffer around planned road')
axes[0].add_patch(plt.Rectangle((xmin-pad,ymin-pad),xmax-xmin+2*pad,ymax-ymin+2*pad,fill=False,ec='blue',lw=1.5))
handles=[Line2D([],[],color='blue',lw=2,label='Planned road'),
 Line2D([],[],color='grey',lw=1,label='Existing roads'),
 Line2D([],[],color='darkred',ls='--',label='5.5 km buffer'),
 Patch(color='red',alpha=0.7,label=f'At-risk area (buffer minus deforested): {risk.area/1e6:,.0f} km²'),
 Patch(facecolor='none',hatch='///',edgecolor='gold',label=f'At-risk area inside protected forest: {riskP.area/1e6:,.0f} km²'),
 Patch(color='#d2a679',label='Deforested area'),
 Patch(color='#2e8b57',alpha=0.55,label='Protected forest'),
 Line2D([],[],color='black',lw=1.5,label='Rondônia boundary (approx.)')]
fig.legend(handles=handles,loc='lower center',ncol=4,fontsize=10,frameon=True)
fig.suptitle('Predicted deforestation risk from the proposed road in Rondônia, Brazil',fontsize=15)
plt.tight_layout(rect=[0,0.08,1,0.96])
plt.savefig('pred_results/predictedRiskyArea.png',dpi=150)
gpd.GeoDataFrame({'type':['risk','risk_in_protected']},geometry=[risk,riskP],crs=crs).to_file('pred_results/predictedRiskyArea.geojson')
