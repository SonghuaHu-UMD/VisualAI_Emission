import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime
import requests
import json
import seaborn as sns
import numpy as np
from tqdm import tqdm
import glob
import ast
import matplotlib as mpl
import contextily as ctx
import os
from scipy.ndimage.filters import gaussian_filter1d
from scipy.spatial import Voronoi
import shapely
from shapely import geometry, ops

pd.options.mode.chained_assignment = None


def split_data_yolo(count_df, v_n):
    count_df[[v_n + '_car', v_n + '_truck', v_n + '_bus', v_n + '_cycle']] = pd.DataFrame(
        count_df[v_n].str.strip('[]').str.split(',').to_list()).astype(int)
    count_df[v_n + '_total'] = count_df[[v_n + '_car', v_n + '_truck', v_n + '_bus', v_n + '_cycle']].sum(axis=1)
    count_df[v_n + '_total_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_total'].diff(-1) * (-1)
    count_df[v_n + '_car_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_car'].diff(-1) * (-1)
    count_df[v_n + '_truck_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_truck'].diff(-1) * (-1)
    count_df[v_n + '_bus_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_bus'].diff(-1) * (-1)
    count_df[v_n + '_cycle_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_cycle'].diff(-1) * (-1)
    return count_df


def split_data_own(count_df, v_n):
    count_df[[v_n + '_car', v_n + '_ptruck', v_n + '_ltruck', v_n + '_htruck', v_n + '_sbus',
              v_n + '_bus', v_n + '_mcycle', v_n + '_truck']] = pd.DataFrame(
        count_df[v_n].str.strip('[]').str.split(',').to_list()).astype(int)
    count_df[v_n + '_total'] = count_df[[v_n + '_car', v_n + '_ptruck', v_n + '_ltruck', v_n + '_htruck', v_n + '_sbus',
                                         v_n + '_bus', v_n + '_mcycle', v_n + '_truck']].sum(axis=1)
    count_df[v_n + '_total_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_total'].diff(-1) * (-1)
    count_df[v_n + '_car_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_car'].diff(-1) * (-1)
    count_df[v_n + '_ptruck_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_ptruck'].diff(-1) * (-1)
    count_df[v_n + '_ltruck_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_ltruck'].diff(-1) * (-1)
    count_df[v_n + '_htruck_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_htruck'].diff(-1) * (-1)
    count_df[v_n + '_sbus_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_sbus'].diff(-1) * (-1)
    count_df[v_n + '_bus_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_bus'].diff(-1) * (-1)
    count_df[v_n + '_mcycle_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_mcycle'].diff(-1) * (-1)
    count_df[v_n + '_truck_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_truck'].diff(-1) * (-1)
    return count_df


def split_data_by_type(count_df, y_var='peds', f_name='car_st', v_name='person'):
    count_df[y_var] = count_df[y_var].astype(str)
    count_df[f_name + '_i'] = count_df[y_var].str.find(v_name)
    count_df[f_name] = count_df[y_var].apply(
        lambda st: st[(st.find(v_name) + len(v_name) + 3):(st.find(v_name) + len(v_name) + 6)])
    count_df[f_name] = count_df[f_name].str.extract('(\d+)', expand=False).astype(float)
    count_df.loc[count_df[f_name + '_i'] == -1, f_name] = 0
    count_df[f_name] = count_df[f_name].fillna(0)
    count_df = count_df.drop(f_name + '_i', axis=1)
    return count_df


# Get volume data: ground truth
'''
traffic_volume_raw = pd.read_csv(r'D:\\NY_Emission\Video_Process\Automated_Traffic_Volume_Counts.csv')
traffic_volume = traffic_volume_raw[(traffic_volume_raw['Yr'] == 2019)].reset_index(drop=True)
traffic_volume['SegmentID'] = traffic_volume['SegmentID'].astype(int).astype(str).apply(lambda x: x.zfill(7))
traffic_volume['Yr'] = traffic_volume['Yr'].astype(str)
for kk in ['M', 'D', 'HH', 'MM']:
    traffic_volume[kk] = traffic_volume[kk].astype(int).astype(str).apply(lambda x: x.zfill(2))
traffic_volume['Datehour'] = pd.to_datetime(traffic_volume[['Yr', 'M', 'D', 'HH']].apply(' '.join, 1),
                                            format='%Y %m %d %H')
traffic_volume.to_pickle(r'D:\\NY_Emission\Video_Process\traffic_volume.pkl')
'''

traffic_volume = pd.read_pickle(r'D:\NY_Emission\Video_Process\data_models\traffic_volume.pkl')
traffic_volume_hour = traffic_volume.groupby(['SegmentID', 'Datehour'])['Vol'].sum().reset_index()
traffic_volume_hour['dayofweek'] = traffic_volume_hour['Datehour'].dt.dayofweek
traffic_volume_hour['Hour'] = traffic_volume_hour['Datehour'].dt.hour
traffic_v_mean = traffic_volume_hour.groupby(['SegmentID', 'dayofweek', 'Hour'])['Vol'].median().reset_index()
traffic_v_meana = traffic_v_mean[traffic_v_mean['dayofweek'] == 4]
# traffic_v_meana['Vol'] = traffic_v_meana['Vol'].astype(int)
# print(len(set(traffic_v_mean['SegmentID'])))
traffic_v_meana.groupby(['Hour'])['Vol'].mean().plot()

# read camera data
cams = pd.DataFrame(requests.get(r'https://webcams.nyctmc.org/api/cameras/').json())
url_list = list(cams['imageUrl'])
cams_gpd = gpd.GeoDataFrame(cams, geometry=gpd.points_from_xy(x=cams.longitude, y=cams.latitude))
cams_gpd = cams_gpd.set_crs('EPSG:4326')

# read needed camera
with open(r'D:\NY_Emission\Video_Process\data_models\All_Detector_Raw.json') as f: traffic_count = json.load(f)
traffic_count = pd.DataFrame(traffic_count['dataset']['samples'])
traffic_count['name'] = traffic_count['name'].str.replace('.jpg', '')
cams_gpd = cams_gpd[cams_gpd['id'].isin(traffic_count['name'])]

'''
# read video count data
all_files = glob.glob(os.path.join(r'D:\\NY_Emission\Video_Process\Results_2023_12_02', "*.csv"))
count_df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
count_df['id'] = count_df['file'].str.replace('.mp4', '')
count_df = count_df.drop('Unnamed: 0', axis=1)
count_df = count_df[count_df['id'].isin(cams_gpd['id'])].reset_index(drop=True)
for kk in ['c1', 'c2', 'c3', 'c4']: count_df = split_data_own(count_df, kk)

# count_df_e = count_df[count_df['file'] == '6b81f2fd-acf3-4af2-adc1-42250156ece8.mp4']
# count_df_e.to_csv(r'count_df_e.csv')

# get timestamp for each frame
all_files = glob.glob(os.path.join(r'D:\\NY_Emission\Video_Process\file_names_all', "*.csv"))
all_files = [var for var in all_files if '12_05' in var]
times_df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
times_df = times_df.drop('Unnamed: 0', axis=1)
times_df = times_df[times_df['ID'].isin(cams_gpd['id'])].reset_index(drop=True)
times_df['file'] = times_df['ID'] + '.mp4'
times_df['Datetime'] = pd.to_datetime(times_df['Datetime'])
times_df['Date'] = times_df['Datetime'].dt.date
times_df['Hour'] = times_df['Datetime'].dt.hour
times_df = times_df.sort_values(by=['ID', 'Timestamp']).reset_index(drop=True)
times_df['frame'] = times_df.groupby(['file', 'Date', 'Hour']).cumcount() + 1
times_df = times_df[['file', 'frame', 'Date', 'Hour', 'Datetime']]

# Merge
count_df['Date_hour'] = count_df['Date_hour'] + '_00_00'
count_df['Date_hour'] = pd.to_datetime(count_df['Date_hour'], format='%Y_%m_%d_%H_%M_%S')
count_df['Date'] = count_df['Date_hour'].dt.date
count_df['Hour'] = count_df['Date_hour'].dt.hour
count_df = count_df.merge(times_df, on=['file', 'frame', 'Date', 'Hour'], how='left')
# count_df['models'] = count_df['models'].apply(lambda x: ast.literal_eval(x))
count_df['peds'] = count_df['peds'].apply(lambda x: ast.literal_eval(x))
# Output
count_df.to_pickle(r'D:\\NY_Emission\Video_Process\count_df_2023_12_02.pkl')
'''

# Read and check information details
count_df = pd.read_pickle(r'D:\NY_Emission\Video_Process\count_df_new_final.pkl')
# count_df = count_df[count_df['similar'] != 1].reset_index(drop=True)
# Get count of peds
count_df = split_data_by_type(count_df, y_var='peds', f_name='peds_count', v_name='person')

### Plot: time series and spatial distribution
# Plot daily time series: by counter
p_c = ['c1_total_s', 'c2_total_s', 'c3_total_s', 'c4_total_s']
ct_weight = count_df.groupby('Hour')['file'].count() / max(count_df.groupby('Hour')['file'].count())
ct_sum = count_df.groupby('Hour')[p_c].sum().div(ct_weight, axis=0) / len(set(count_df['file']))
# ct_sum = count_df.groupby('Hour')[p_c].sum()/ len(set(count_df['file']))
fig, ax = plt.subplots(figsize=(6, 5))
# for kk in p_c:
#     ct_sum[p_c] = gaussian_filter1d(ct_sum[p_c], sigma=0.5)
ct_sum[p_c].plot(ax=ax, marker='o', markersize=4, cmap='tab20', alpha=0.8, lw=2)
# ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
plt.ylabel(r'Vehicle count')
plt.xlabel(r'Hour')
plt.legend(['C1', 'C2', 'C3', 'C4'], ncol=2, loc=2)
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\Counter_compare_new.pdf')
plt.close()

# Plot daily time series: by each type
p_c = ['c3_car_s', 'c3_ptruck_s', 'c3_htruck_s', 'c3_ltruck_s', 'c3_sbus_s', 'c3_bus_s', 'c3_mcycle_s', 'c3_truck_s']
new_n = ['Passenger Car', 'Passenger Truck', 'Light Truck', 'Heavy Truck', 'Transit Bus', 'School Bus', 'Motorcycle']
new_nn = ['P_Car', 'P_Truck', 'SU_Truck', 'C_Truck', 'T_Bus', 'S_Bus', 'M_Cycle']
ct_weight = count_df.groupby(['file', 'Hour'])['frame'].count() / 3600
ct_sum = count_df.groupby(['file', 'Hour'])[p_c].sum().div(ct_weight, axis=0).reset_index()
ct_sum_h = ct_sum.groupby(['Hour'])[p_c].mean()
ct_peds = (count_df.groupby(['file', 'Hour'])['peds_count'].mean() * 3600).reset_index()
ct_sump_h = ct_peds.groupby(['Hour'])['peds_count'].mean().reset_index()
ct_sum_h = ct_sum_h.merge(ct_sump_h, on='Hour')
ct_sum_h.set_index('Hour', inplace=True)
ct_sum_h['P_Car'] = ct_sum_h['c3_car_s']
ct_sum_h['P_Truck'] = ct_sum_h['c3_ptruck_s']  # + ct_sum_h['c3_truck_s']
ct_sum_h['SU_Truck'] = ct_sum_h['c3_ltruck_s']
ct_sum_h['C_Truck'] = ct_sum_h['c3_htruck_s']
ct_sum_h['T_Bus'] = ct_sum_h['c3_bus_s']
ct_sum_h['S_Bus'] = ct_sum_h['c3_sbus_s']
ct_sum_h['M_Cycle'] = ct_sum_h['c3_mcycle_s'] * 4
ct_sum_h = ct_sum_h[new_nn]
print(ct_sum_h.sum() / ct_sum_h.sum().sum())
ct_sum_h['Total'] = ct_sum_h.sum(axis=1)

fig, ax = plt.subplots(figsize=(6.75, 5), sharex=True)
ct_sum_h_s = pd.read_csv(r'D:\NY_Emission\Figure\ct_sum_h_s.csv', index_col=0)
for kk in ct_sum_h_s.columns:
    ct_sum_h_s[kk] = gaussian_filter1d(ct_sum_h_s[kk], sigma=1)
ct_sum_h_s.plot(ax=ax, marker='o', markersize=4, cmap='tab20', alpha=0.8, lw=2, subplots=True, layout=(2, 3),
                legend=False, title=['P_Car', 'P_Truck', 'SU_Truck', 'T_Bus', 'S_Bus', 'M_Cycle'])
plt.subplots_adjust(top=0.922, bottom=0.117, left=0.09, right=0.985, hspace=0.577, wspace=0.283)
plt.savefig(r'D:\NY_Emission\Figure\CarType_daily_wd.pdf')
plt.close()

# Output ped and car volume of each camera
ct_sum.merge(ct_peds, on=['file', 'Hour']).to_csv(r'D:\NY_Emission\Figure\peds_car_new.csv')

# Plot spatial camera count by type
county_us = gpd.read_file(
    r'G:\Data\Dewey\SAFEGRAPH\Open Census Data\Census Website\2019\nhgis0011_shape\\US_county_2019.shp')
count_df = pd.read_pickle(r'D:\NY_Emission\Video_Process\count_df.pkl')
# count_df = split_data_by_type(count_df, y_var='peds', f_name='peds_count', v_name='person')
ct_weight = count_df.groupby(['file', 'Hour'])['frame'].count() / 3600
ct_sum = count_df.groupby(['file', 'Hour'])[['c3_car_s', 'c3_truck_s', 'c3_bus_s', 'c3_cycle_s'
                                             ]].sum().div(ct_weight, axis=0).reset_index()
ct_sum['id'] = ct_sum['file'].str.replace('.mp4', '')
cams_gpd_all = cams_gpd.merge(ct_sum, on='id')
cams_gpd_all = cams_gpd_all[cams_gpd_all['area'] == 'Manhattan']
cams_gpd_all = cams_gpd_all.to_crs('EPSG:4326')

for c_type in ['c3_car_s', 'c3_truck_s', 'c3_bus_s']:
    for tt in [3, 8, 16]:
        each_count = cams_gpd_all[cams_gpd_all['Hour'] == tt]
        # Generate a Thiessen Polygon
        x = each_count.geometry.x.values
        y = each_count.geometry.y.values
        coords = np.vstack((x, y)).T
        vor = Voronoi(coords)
        lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
        polys = shapely.ops.polygonize(lines)
        voronois = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys))
        voronois = voronois.set_crs('EPSG:4326')

        # Cut by Manhattan
        county_mt = county_us[county_us['NAMELSAD'] == 'New York County']
        county_mt = county_mt.to_crs('EPSG:32618')
        county_mt['geometry'] = county_mt['geometry'].buffer(100)
        county_mt = county_mt.to_crs('EPSG:4326')
        voronois = gpd.overlay(voronois, county_mt)
        voronois = voronois.reset_index()

        fig, ax = plt.subplots(figsize=(3.5, 7))
        mpl.rcParams['text.color'] = 'white'
        each_count.plot(column=c_type, cmap='RdYlGn_r', scheme="natural_breaks", k=6,
                        markersize=100 * (each_count[c_type] / max(cams_gpd_all[c_type])), ax=ax, alpha=0.9,
                        legend=True, legend_kwds={'labelcolor': 'white', "fmt": "{:.0f}", 'ncol': 1, 'title': 'Volume',
                                                  'loc': 'upper left', 'frameon': True, 'facecolor': 'k',
                                                  'edgecolor': 'k', 'framealpha': 0.5})
        mpl.rcParams['text.color'] = 'k'
        # voronois.boundary.plot(ax=ax, color='gray', lw=0.05, alpha=0.5)
        ctx.add_basemap(ax, crs=each_count.crs, source=ctx.providers.CartoDB.DarkMatter, alpha=0.9)
        plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
        # plt.tight_layout()
        plt.axis('off')
        # plt.savefig(r'D:\NY_Emission\Figure\LVolume_camera.pdf')
        plt.savefig(r'D:\NY_Emission\Figure\spatial\CamerVolume_%s_%s.pdf' % (c_type, tt))
        plt.close()

# Volume validation: find the closest link for each camera and each hour
p_c = ['c1_total_s', 'c2_total_s', 'c3_total_s', 'c4_total_s']
ct_weight = count_df.groupby(['file', 'Hour'])['frame'].count() / 3600
ct_sum = count_df.groupby(['file', 'Hour'])[p_c].sum().div(ct_weight, axis=0).reset_index()
ct_sum['id'] = ct_sum['file'].str.replace('.mp4', '')
cams_gpd = cams_gpd.merge(ct_sum, on='id')
cams_gpd = cams_gpd.to_crs('EPSG:3857')

lion = gpd.read_file(r'D:\NY_Emission\Shp\AADT_New\Dynamic_count.shp')
lion = lion.drop_duplicates(['SegmentID']).reset_index(drop=True)
# traffic_v_mean_all.columns = ['SegmentID', 'Vol18']
# lion = lion.merge(traffic_v_mean_all, on='SegmentID')
lion = lion.to_crs('EPSG:3857')
lion['ref_points'] = 0
lion['is_match'] = 0
matched = pd.DataFrame()
for jj in tqdm(range(0, len(cams_gpd))):
    tem_point = cams_gpd.loc[jj,]
    # Calculate the distance
    lion.loc[:, 'ref_points'] = tem_point.geometry
    lion.loc[:, "snap_dist"] = lion.geometry.distance(gpd.GeoSeries(lion['ref_points'], crs='EPSG:3857'))
    temp_map = lion[(lion['snap_dist'] < 10)]
    if len(temp_map) > 0:
        # break
        # select the nearest
        temp_map = temp_map.merge(traffic_v_meana, on='SegmentID')  # detector
        temp_map = temp_map.merge(ct_sum[ct_sum['id'] == tem_point['id']], on='Hour')  # camera
        matched = pd.concat([matched, temp_map])

# Plot comparison
matched['abs_diff'] = abs(matched['total_final'] - matched['Vol_y'])
needid = matched.groupby(['id'])['diff_final'].mean().reset_index()

matched_n = matched.copy()
print(matched_n[['total_final', 'Vol_y']].corr())
print(matched_n['diff_final'].mean())

fig, ax = plt.subplots(figsize=(5, 5))
sns.regplot(data=matched_n, x='Vol_y', y='total_final', ax=ax, scatter_kws={'alpha': 0.4, 's': 10})
ax.plot([0, 2500], [0, 2500], '--', lw=3, color='k')
plt.ylabel('Hourly volume (Camera)')
plt.xlabel('Hourly volume (ATR)')
plt.tight_layout()
plt.text(0.7, 0.96, r'$\rho = $' + str(round(matched_n[['total_final', 'Vol_y']].corr().values[1][0], 2)),
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.text(0.7, 0.9, 'MAPE = ' + str(round(matched_n['diff_final'].median() * 100, 2)) + ' %',
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.savefig(r'D:\NY_Emission\Figure\volume_compare.pdf')
plt.savefig(r'D:\NY_Emission\Figure\volume_compare.png',dpi=1000)

sns.set_palette("tab20")
fig, ax = plt.subplots(figsize=(6, 5))
matched_n['diff_final'] = matched_n['diff_final'] * 100
sns.boxplot(data=matched_n, x="Hour", y="diff_final")
ax.xaxis.set_ticks(np.arange(0, 24, 4))
plt.ylabel('MAPE (%)')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\volume_compare_hour_mape.pdf')

fig, ax = plt.subplots(figsize=(6, 5))
matched_n.groupby('Hour')['total_final'].mean().plot(ax=ax, marker='o', markersize=4, lw=2)
matched_n.groupby('Hour')['Vol_y'].mean().plot(ax=ax, marker='o', markersize=4, lw=2)
plt.ylabel(r'Vehicle count')
plt.xlabel(r'Hour')
plt.legend(['Camera', 'ATR'], ncol=1, loc=2)
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\volume_compare_hour_plot.pdf')

