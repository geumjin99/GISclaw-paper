"""Correct: the coverage (96.77 %) is held in a dict together with booleans and written as a
metric / value row of a CSV.
"""
import geopandas as gpd, pandas as pd, numpy as np
st=gpd.read_file('dataset/Streets.shp'); nb=gpd.read_file('dataset/Neighborhoods.shp'); zn=gpd.read_file('dataset/Zoning.shp')
print(st.columns.tolist())
print(st[['SEGMENT_TY','ARTERIAL_C']].value_counts().head(15))
print(nb[nb.S_HOOD.fillna('').str.contains('Capitol|Broadway|Stevens|Madison|Miller|Montlake|Pike|First Hill|Central',case=False)][['S_HOOD','L_HOOD']])
print(nb.L_HOOD.unique())

art=st[(st.SEGMENT_TY==1)&(st.ARTERIAL_C==1)].copy()
print(len(art))
pat='MADISON|10TH AVE E|23RD AVE|24TH AVE E|TURNER'
print(art[art.ORD_STNAME.str.contains(pat)].ORD_STNAME.value_counts())
print(nb[nb.L_HOOD=='CAPITOL HILL'][['S_HOOD','AREA']])

art['NAME']=art.ORD_STNAME.str.strip()
sec={'E MADISON ST':'East','MADISON ST':'East','10TH AVE E':'West','23RD AVE':'Central','23RD AVE E':'Central','24TH AVE E':'Central','TURNER WAY E':'Central'}
routes=art[art.NAME.isin(sec)].copy(); routes['SECTION']=routes.NAME.map(sec)
routes['len_ft']=routes.length
print(routes.groupby('SECTION').agg(n=('NAME','size'),len_mi=('len_ft',lambda s:s.sum()/5280)))
for s,g in routes.groupby('SECTION'): print(s,g.total_bounds.round(0))
ch=nb[nb.L_HOOD=='CAPITOL HILL']; print(ch.total_bounds.round(0))

import networkx as nx
from shapely.ops import unary_union
def graph(gdf):
    G=nx.Graph()
    for i,geom in zip(gdf.index,gdf.geometry):
        parts=geom.geoms if geom.geom_type=='MultiLineString' else [geom]
        for p in parts:
            a=tuple(np.round(p.coords[0],0)); b=tuple(np.round(p.coords[-1],0))
            G.add_edge(a,b,weight=p.length,idx=i)
    return G
Gr=graph(routes)
comps=list(nx.connected_components(Gr)); print('components',len(comps),[len(c) for c in comps])
buf=gpd.GeoDataFrame(geometry=[routes.geometry.buffer(2640).union_all()],crs=st.crs)
chu=ch.geometry.union_all()
cov=chu.intersection(buf.geometry[0]).area/chu.area
print('Capitol Hill area covered:',round(cov*100,1),'%')
for _,r in ch.iterrows(): print(r.S_HOOD, round(r.geometry.intersection(buf.geometry[0]).area/r.geometry.area*100,1))

small=min(comps,key=len); big=max(comps,key=len)
print(routes.loc[[d['idx'] for u,v,d in Gr.edges(data=True) if u in small],'NAME'].unique())
Ga=graph(art)
print('nodes in Ga small/big', sum(n in Ga for n in small), sum(n in Ga for n in big))
# multi-source shortest path: add virtual source/target
H=Ga.copy()
for n in small: H.add_edge('S',n,weight=0)
for n in big: H.add_edge('T',n,weight=0)
path=nx.shortest_path(H,'S','T',weight='weight')
edges=[H[u][v]['idx'] for u,v in zip(path[1:-2],path[2:-1])]
conn=art.loc[edges].copy(); print(conn[['NAME','len_ft' if 'len_ft' in conn else 'GIS_SEG_LE']]); print('connector ft',conn.length.sum())

zn['geometry']=zn.geometry.make_valid()
print(zn.CLASS_DESC.value_counts())
study=chu.union(nb[nb.S_HOOD.isin(['Pike-Market','Central Business District','First Hill'])].geometry.union_all()).buffer(2640)
znS=zn[zn.intersects(study)].copy(); print(len(znS))
print(znS.groupby('CLASS_DESC').geometry.apply(lambda g:g.area.sum()/27878400).round(2))

from shapely.geometry import box
w={'Downtown':1.0,'Major Institutions':0.8,'Neighborhood/Commercial':0.7,'Residential/Commercial':0.6,'Multi-Family':0.5,'Manufacturing/Industrial':0.4,'Master Planned Community':0.5,'Single Family':0.1}
znS['dw']=znS.CLASS_DESC.map(w)
print('zones without class weight:',znS.dw.isna().sum())
x0,y0,x1,y1=study.bounds; cs=500
cells=[box(x,y,x+cs,y+cs) for x in np.arange(x0,x1,cs) for y in np.arange(y0,y1,cs)]
grid=gpd.GeoDataFrame(geometry=cells,crs=st.crs); grid=grid[grid.intersects(study)].reset_index(drop=True); grid['gid']=grid.index
ov=gpd.overlay(grid,znS[znS.dw.notna()][['dw','CLASS_DESC','geometry']],how='intersection',keep_geom_type=True)
ov['a']=ov.area
d=ov.groupby('gid').apply(lambda g:(g.dw*g.a).sum()/g.a.sum())
grid['demand']=grid.gid.map(d)
print(len(grid), grid.demand.isna().sum(), grid.demand.describe().round(2))

from libpysal.weights import DistanceBand
from esda.getisord import G_Local
gv=grid[grid.demand.notna()].copy()
cent=np.column_stack([gv.centroid.x,gv.centroid.y])
W=DistanceBand(cent,threshold=1100,binary=True,silence_warnings=True)
gi=G_Local(gv.demand.values,W,star=True,permutations=999,seed=1)
gv['GiZ']=gi.Zs; gv['GiP']=gi.p_sim
gv['hotspot']=np.where((gv.GiZ>1.96)&(gv.GiP<0.05),'Hot',np.where((gv.GiZ<-1.96)&(gv.GiP<0.05),'Cold','Not significant'))
print(gv.hotspot.value_counts())
hot=gv[gv.hotspot=='Hot']; hotU=hot.geometry.union_all()
print('hot cells within half-mile of routes:', round(hot.centroid.within(buf.geometry[0]).mean()*100,1),'%')

print(pd.Series(gi.Zs).describe().round(2)); print(pd.Series(gi.p_sim).describe().round(3)); print(pd.Series(gi.p_norm).describe().round(3))
print(W.cardinalities and pd.Series(W.cardinalities).describe())

gi=G_Local(gv.demand.values,W,transform='B',star=True,permutations=999,seed=1)
print(pd.Series(gi.Zs).describe().round(2)); print(pd.Series(gi.z_sim).describe().round(2))
gv['GiZ']=gi.Zs; gv['GiP']=gi.p_sim
gv['hotspot']=np.where((gv.GiZ>1.96)&(gv.GiP<0.05),'Hot',np.where((gv.GiZ<-1.96)&(gv.GiP<0.05),'Cold','Not significant'))
print(gv.hotspot.value_counts())
hot=gv[gv.hotspot=='Hot']
print('hot cells within half-mile of routes:', round(hot.centroid.within(buf.geometry[0]).mean()*100,1),'%')

stS=st[st.intersects(study)&(st.SEGMENT_TY==1)].copy()
print(stS.ARTERIAL_C.value_counts())

from sklearn.preprocessing import minmax_scale
stS['NAME']=stS.ORD_STNAME.str.strip()
stS['s_traffic']=np.where(stS.ARTERIAL_C==1,1.0,0.2)   # arterial = busier street (priority per City policy)
mid=stS.geometry.interpolate(0.5,normalized=True)
j=gpd.sjoin(gpd.GeoDataFrame(geometry=mid,crs=st.crs),gv[['GiZ','geometry']],how='left',predicate='within')
j=j[~j.index.duplicated()]
stS['GiZ']=j.GiZ
print('segments without Gi value:',stS.GiZ.isna().sum())
stS['s_demand']=minmax_scale(stS.GiZ.fillna(stS.GiZ.min()).clip(-3,3))
emp=znS[znS.CLASS_DESC.isin(['Downtown','Major Institutions','Neighborhood/Commercial','Residential/Commercial'])].geometry.union_all()
stS['d_emp']=mid.distance(emp)
stS['s_emp']=1-np.clip(stS.d_emp/2640,0,1)
stS['suit']=0.4*stS.s_traffic+0.4*stS.s_demand+0.2*stS.s_emp
stS['proposed']=stS.index.isin(routes.index); stS['connector']=stS.index.isin(conn.index)
print(stS.suit.describe().round(3))
print(stS.groupby(['proposed','ARTERIAL_C']).suit.mean().round(3))

stS['s_demand']=(stS.GiZ.clip(-3,3)+3)/6   # NaN stays NaN
stS['suit']=0.4*stS.s_traffic+0.4*stS.s_demand+0.2*stS.s_emp
stS['suit_missing']=stS.suit.isna()
print('missing suit:',stS.suit_missing.sum())
print(stS.groupby(['proposed']).suit.mean().round(3), 'connector mean', stS[stS.connector].suit.mean().round(3))
net=pd.concat([routes.assign(ROLE='City proposed'),conn.assign(SECTION='Connector',ROLE='Proposed connector')])
bufN=net.geometry.buffer(2640).union_all()
studyClip=lambda g: g.intersection(study)
metrics={}
metrics['route_miles_city']=routes.length.sum()/5280; metrics['route_miles_with_connector']=net.length.sum()/5280
metrics['network_components_city']=len(comps); metrics['network_components_with_connector']=nx.number_connected_components(graph(net))
metrics['capitol_hill_area_pct_halfmile']=chu.intersection(bufN).area/chu.area*100
for _,r in ch.iterrows(): metrics[f'pct_{r.S_HOOD}']=r.geometry.intersection(bufN).area/r.geometry.area*100
empS=emp.intersection(study); metrics['employment_zone_area_pct_halfmile']=empS.intersection(bufN).area/empS.area*100
dt=nb[nb.S_HOOD.isin(['Central Business District','First Hill','Pike-Market'])]
for _,r in dt.iterrows(): metrics[f'route_reaches_{r.S_HOOD}']=bool(net.intersects(r.geometry).any())
metrics['demand_hot_cells_pct_halfmile']=hot.centroid.within(bufN).mean()*100
for k,v in metrics.items(): print(k, round(v,1) if isinstance(v,float) else v)

import os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.makedirs('pred_results',exist_ok=True)
gp='pred_results/capitol_hill_bike_routes.gpkg'
keep=['NAME','SECTION','ROLE','ARTERIAL_C','SEGMENT_TY','GIS_SEG_LE','geometry']
art[['NAME','ARTERIAL_C','SEGMENT_TY','GIS_SEG_LE','geometry']].to_file(gp,layer='arterial_streets',driver='GPKG')
net[keep].to_file(gp,layer='proposed_bike_routes',driver='GPKG')
gpd.GeoDataFrame({'dist_ft':[2640]},geometry=[bufN],crs=st.crs).to_file(gp,layer='half_mile_buffer',driver='GPKG')
gv[['gid','demand','GiZ','GiP','hotspot','geometry']].to_file(gp,layer='demand_hotspots_gi',driver='GPKG')
stS[['NAME','ARTERIAL_C','s_traffic','GiZ','s_demand','d_emp','s_emp','suit','suit_missing','proposed','connector','geometry']].to_file(gp,layer='street_suitability',driver='GPKG')
pd.DataFrame([(k,round(float(v),2) if not isinstance(v,bool) else v) for k,v in metrics.items()],columns=['metric','value']).to_csv('pred_results/accessibility_metrics.csv',index=False)
net.drop(columns='geometry')[['NAME','SECTION','ROLE','GIS_SEG_LE']].to_csv('pred_results/proposed_routes.csv',index=False)

fig,ax=plt.subplots(1,2,figsize=(16,10))
xmin,ymin,xmax,ymax=bufN.union(chu).bounds
for a in ax: a.set_xlim(xmin-1500,xmax+1500); a.set_ylim(ymin-1500,ymax+1500); a.set_axis_off()
cmap={'Hot':'#d7301f','Cold':'#4575b4','Not significant':'#eeeeee'}
gv.plot(ax=ax[0],color=gv.hotspot.map(cmap),edgecolor='none')
ch.boundary.plot(ax=ax[0],color='k',lw=1)
gpd.GeoSeries([bufN]).boundary.plot(ax=ax[0],color='purple',ls='--',lw=1.2)
colors={'East':'#1b9e77','West':'#d95f02','Central':'#7570b3','Connector':'#e7298a'}
for s,g in net.groupby('SECTION'): g.plot(ax=ax[0],color=colors[s],lw=3,label=s)
ax[0].legend(loc='lower left',title='Route section'); ax[0].set_title('Demand hotspots (Gi*, zoning-based proxy), proposed routes & ½-mile coverage')
stS[stS.suit.notna()].plot(ax=ax[1],column='suit',cmap='RdYlGn',lw=1,legend=True,legend_kwds={'label':'Suitability (0-1)','shrink':0.6})
net.plot(ax=ax[1],color='k',lw=0.8); ch.boundary.plot(ax=ax[1],color='grey',lw=0.8)
ax[1].set_title('Weighted suitability: 0.4 traffic(arterial) + 0.4 demand + 0.2 employment proximity')
plt.tight_layout(); plt.savefig('pred_results/bike_route_analysis_map.png',dpi=150); plt.close()
print(os.listdir('pred_results'))