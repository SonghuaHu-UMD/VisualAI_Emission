import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import curve_fit
import seaborn as sns
import datetime
import matplotlib.dates as mdates


def dens_spd_func(x, ffs, k_critical, mm):  # fundamental diagram model (density-speed function)
    x_over_k = x / k_critical
    dominator = 1 + np.power(x_over_k, mm)
    order = 2 / mm
    return ffs / np.power(dominator, order)


def bpr_function(x, ffs, alpha, beta):
    return ffs / (1 + alpha * np.power(x, beta))


# Read link
link = gpd.read_file(r'D:\NY_Emission\Shp\osmdta_ritis.shp')
link = link.rename({'from_node_': 'from_node_id', 'link_type_': 'link_type_name'}, axis=1)
link = link.to_crs(epsg=3857)
link['linkLength'] = link.geometry.length  # meter

# Read speed
speed = pd.read_csv(r'D:\NY_Emission\Speed\NY_TT\NY_TT.csv')
speed['measurement_tstamp'] = pd.to_datetime(speed['measurement_tstamp'])
speed = speed[speed['measurement_tstamp'].dt.date == datetime.date(2023, 12, 5)].reset_index(drop=True)
# speed.groupby(['measurement_tstamp'])['speed'].mean().plot()
speed['Hour'] = speed['measurement_tstamp'].dt.hour
speed = speed.groupby(['tmc_code', 'Hour'])['speed'].mean().reset_index()

# Read volume
assign_all = pd.read_pickle(r'D:\NY_Emission\ODME_NY\Simulation_outcome\assign_all_%s.pkl' % '')

# Match speed-density relationships
aadt = link.merge(speed[['tmc_code', 'speed', 'Hour']], on=['tmc_code'], how='left')
aadt.rename({'Hour': 'hour'}, axis=1, inplace=True)
aadt = aadt.merge(assign_all[['from_node_id', 'to_node_id', 'volume_hourly', 'hour']],
                  on=['from_node_id', 'to_node_id', 'hour'], how='left')
aadt['density_hourly'] = aadt['volume_hourly'] / (aadt['linkLength'] * aadt['lanes'])
# aadt['density_hourly'] = aadt['volume_hourly'] / aadt['lanes']
aadt[aadt['hour'].isin([8, 17])].groupby('link_type_name')['density_hourly'].mean()

# Curve fit
need_sds = pd.DataFrame()
paras = pd.DataFrame()
fig, ax = plt.subplots(figsize=(4.5, 4))
for kk in tqdm(list(set(aadt['tmc_code']))):
    data_ft = aadt[(aadt['volume_hourly'] > 0) & (aadt['speed'] > 0) & (aadt['tmc_code'] == kk)]
    x = np.array(data_ft['density_hourly'])
    y = np.array(data_ft['speed'])
    roadtype = list(set(data_ft['link_type_name']))
    if len(roadtype) > 0:
        roadtype = roadtype[0]
    else:
        roadtype = ''
    if roadtype in ['motorway']:
        clor = '#00A087FF'
    elif roadtype in ['trunk', 'primary']:
        clor = '#E64B35FF'
    elif roadtype in ['secondary']:
        clor = '#6C81B5'
    elif roadtype in ['residential']:
        clor = '#D175AB'
    if len(x) > 10:
        # popt, pcov = curve_fit(dens_spd_func, x, y, bounds=[[20, 1, 0], [200, 80, 2]], maxfev=5000)
        if roadtype == 'motorway':
            popt, pcov = curve_fit(dens_spd_func, x, y, bounds=[[60, 1, 0], [200, 80, 3]], maxfev=5000)
        elif roadtype in ['trunk', 'primary']:
            popt, pcov = curve_fit(dens_spd_func, x, y, bounds=[[50, 1, 0], [200, 60, 3]], maxfev=5000)
        elif roadtype in ['secondary']:
            popt, pcov = curve_fit(dens_spd_func, x, y, bounds=[[40, 1, 0], [200, 50, 3]], maxfev=5000)
        else:
            popt, pcov = curve_fit(dens_spd_func, x, y, bounds=[[30, 1, 0], [200, 30, 3]], maxfev=5000)
        xvals = np.sort(x)

        ax.plot(x, y, '*', c=clor, markersize=1, alpha=0.3)
        ax.plot(xvals, dens_spd_func(xvals, *popt), '-', c=clor, lw=2, alpha=0.6)
        need_sd = pd.DataFrame([x, y, dens_spd_func(xvals, *popt)]).T
        need_sd.columns = ['x', 'y', 'y_pre']
        need_sd['rtype'] = roadtype
        need_sd['tmc_code'] = kk
        need_sds = pd.concat([need_sds, need_sd], ignore_index=True)

        paras0 = pd.DataFrame(popt).T
        paras0.columns = ['ffs', 'k_critical', 'mm']
        paras0['rtype'] = roadtype
        paras0['tmc_code'] = kk
        paras = pd.concat([paras0, paras], ignore_index=True)

plt.plot([], label="motorway", color="#00A087FF", alpha=0.6)
plt.plot([], label="primary", color="#E64B35FF", alpha=0.6)
plt.plot([], label="secondary", color="#6C81B5", alpha=0.6)
plt.plot([], label="residential", color="#D175AB", alpha=0.6)
plt.legend(loc='upper right')
plt.xlabel('Density' + r'$(veh*mile^{-1}*lane^{-1})$')
plt.ylabel('Speed(mph)')
plt.xlim([0, 150])
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\Speed_density.pdf')
plt.savefig(r'D:\NY_Emission\Figure\Speed_density.png', dpi=600)

# Output parameters for later use
paras.to_csv(r'D:\NY_Emission\Figure\VSD_Parameter.csv', index=False)
paras.groupby('rtype')[['ffs', 'k_critical', 'mm']].mean()
