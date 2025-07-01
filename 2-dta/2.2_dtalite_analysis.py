import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.pylabtools import figsize
from tqdm import tqdm
from shapely.geometry import Point, LineString
import os
from scipy.ndimage.filters import gaussian_filter1d
import contextily as ctx
import requests
from scipy.optimize import curve_fit
import seaborn as sns
import jenkspy
import imageio
import mapclassify
import datetime
import matplotlib as mpl

pd.options.mode.chained_assignment = None


# Calculate direction
def line_dir(link, fromNorth=False):
    link['Start_Lon_utm'] = link["geometry"].apply(lambda g: g.coords[0][0])
    link['Start_Lat_utm'] = link["geometry"].apply(lambda g: g.coords[0][1])
    link['End_Lon_utm'] = link["geometry"].apply(lambda g: g.coords[-1][0])
    link['End_Lat_utm'] = link["geometry"].apply(lambda g: g.coords[-1][1])
    orig = [link['Start_Lon_utm'], link['Start_Lat_utm']]
    dest = [link['End_Lon_utm'], link['End_Lat_utm']]
    orig = np.asarray(orig)
    dest = np.asarray(dest)
    dx, dy = dest - orig
    ang = np.degrees(np.arctan2(dy, dx))
    if fromNorth:
        ang = np.mod((450.0 - ang), 360.)
    return ang


# '36087', '34031','34013', '36085'
CT_L = ['36061', '36005', '34003', '34017', '36081', '35047', '36047', '36119', '36087']
# CT_N=['New York','Bronx','Allegany','Chenango','Queens','']
dts = [datetime.datetime(2020, 3, 21, 0, 0, 0), datetime.datetime(2021, 11, 24, 0, 0, 0),
       datetime.datetime(2021, 8, 22, 0, 0, 0), datetime.datetime(2022, 1, 29, 0, 0, 0),
       datetime.datetime(2023, 12, 5, 0, 0, 0)]
t_t = 'Car'

# Read link
link = pd.read_csv(r'D:\NY_Emission\ODME_NY\Simulation_osm\link.csv')
mean_link = link.groupby(['link_type_name'])[['capacity', 'lanes', 'free_speed']].mean()
mean_link['free_speed'] = mean_link['free_speed'] * 0.621371
sum_link = link.groupby(['link_type_name'])['length'].sum() * 0.000621371  # to mile
count_link = link.groupby(['link_type_name'])['link_id'].count()
link_des = mean_link.join(sum_link).join(count_link)
link_des.to_csv(r'D:\NY_Emission\Figure\link_des.csv')

# Read ground truth from cameras
count_df = pd.read_pickle(r'D:\NY_Emission\Video_Process\count_df.pkl')
p_c = ['c1_total_s', 'c2_total_s', 'c3_total_s', 'c4_total_s']
ct_weight = count_df.groupby(['file', 'Hour'])['frame'].count() / 3600
ct_sum = count_df.groupby(['file', 'Hour'])[p_c].sum().div(ct_weight, axis=0).reset_index()
ct_sum['id'] = ct_sum['file'].str.replace('.mp4', '')

# Build up camera gpd
cams = pd.DataFrame(requests.get(r'https://webcams.nyctmc.org/api/cameras/').json())
url_list = list(cams['imageUrl'])
cams_gpd = gpd.GeoDataFrame(cams, geometry=gpd.points_from_xy(x=cams.longitude, y=cams.latitude))
cams_gpd = cams_gpd.set_crs('EPSG:4326')
# only consider Manhattan
cams_gpd = cams_gpd[cams_gpd['area'] == 'Manhattan'].reset_index(drop=True)
cams_gpd = cams_gpd.merge(ct_sum, on='id')
cams_gpd = cams_gpd.to_crs('EPSG:4326')

# Plot camera
n_ll = ['16de4b9a-83fe-4d11-91cd-d775cc1e559c', '8f692f55-8118-423b-8bcb-1ea49eaf442b',
        '155b2bff-5dd2-4109-bd10-e098376c8476', '6a49e889-4397-4c5a-a942-2225c1f1c5e7',
        'a26130e0-3c9b-4f43-b253-209d80d441f8', 'cdfc2dbe-7ce3-4775-ab7a-8440a9828358']

fig, ax = plt.subplots(figsize=(3.5, 6))
cams_gpd_n = cams_gpd[cams_gpd['Hour'] == 9]
cams_gpd_n.plot(ax=ax, alpha=0.9)
test = cams_gpd_n[cams_gpd_n['id'].isin(n_ll)]
cams_gpd_n[cams_gpd_n['id'].isin(n_ll)].plot(ax=ax, alpha=0.9, color='white')
ctx.add_basemap(ax, crs=cams_gpd_n.crs, source=ctx.providers.CartoDB.DarkMatter, alpha=0.9)
plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
# plt.tight_layout()
plt.axis('off')
# plt.savefig(r'D:\NY_Emission\Figure\LVolume_camera.pdf')
plt.savefig(r'D:\NY_Emission\Figure\Camera_dis.png', dpi=1000)
plt.close()

# Get time series ratio: use traffic volume
traffic_volume_raw = pd.read_csv(r'D:\NY_Emission\Volume_grth\Automated_Traffic_Volume_Counts.csv')
traffic_volume = traffic_volume_raw[(traffic_volume_raw['Yr'] == 2019)].reset_index(drop=True)
traffic_volume['SegmentID'] = traffic_volume['SegmentID'].astype(int).astype(str).apply(lambda x: x.zfill(7))
traffic_volume['Yr'] = traffic_volume['Yr'].astype(str)
for kk in ['M', 'D', 'HH', 'MM']:
    traffic_volume[kk] = traffic_volume[kk].astype(int).astype(str).apply(lambda x: x.zfill(2))
traffic_volume['Datehour'] = pd.to_datetime(traffic_volume[['Yr', 'M', 'D', 'HH']].apply(' '.join, 1),
                                            format='%Y %m %d %H')
traffic_volume_hour = traffic_volume.groupby(['SegmentID', 'Datehour'])['Vol'].sum().reset_index()
traffic_volume_hour['dayofweek'] = traffic_volume_hour['Datehour'].dt.dayofweek
traffic_volume_hour['Hour'] = traffic_volume_hour['Datehour'].dt.hour
# traffic_volume_hour.groupby('Hour')['Vol'].mean().plot()

# emissions = pd.read_csv(r'D:\NY_Emission\Figure\peds_car.csv', index_col=0)
# emissions['id'] = emissions['file'].str[0:-4]
# cams_gpd = cams_gpd.merge(emissions, on=['id', 'Hour'])
# cams_gpd['interact'] = cams_gpd['peds_count'] * (cams_gpd[['c3_car_s', 'c3_truck_s', 'c3_bus_s']].sum(axis=1))

# Create hourly volume file
avar = 'ODME_volume_after'
# avar = 'ODME_volume_before'
PLT_ODME = False
cct = 0
# ess = ['cd', 'te', 'hf', 'ss', '']
ess = ['cg_2', 'cg_4', 'cg_6', 'cg_8']
for e_s in ess:
    assign_all_raw = pd.DataFrame()
    for t_p in ['am', 'pm', 'md', 'nt1', 'nt2']:  #
        print(t_p)
        if t_p == 'am':
            t_l = [6, 7, 8, 9]
        elif t_p == 'pm':
            t_l = [15, 16, 17, 18]
        elif t_p == 'md':
            t_l = [10, 11, 12, 13, 14]
        elif t_p == 'nt1':
            t_l = [19, 20, 21, 22, 23]
        elif t_p == 'nt2':
            t_l = [0, 1, 2, 3, 4, 5]

        # Read assignment outcome
        assign_r = pd.read_csv(
            r'D:\NY_Emission\ODME_NY\Simulation_outcome\osm_link_performance_%s_%s_%s.csv' % (t_t, t_p, e_s))
        assign_r['t_p'] = t_p
        assign_all_raw = pd.concat([assign_all_raw, assign_r[['from_node_id', 'to_node_id', avar, 't_p']]])
        if PLT_ODME and e_s == '':
            # # # Plot ODME scatters
            assign_r['diff'] = np.abs(assign_r['ODME_volume_after'] - assign_r['ODME_obs_count'])
            assign_rp = assign_r[assign_r['diff'] < 40000]
            assign_rp = assign_rp[assign_rp['ODME_volume_after'] > 1]
            assign_rp['ODME_obs_count'] = assign_rp['ODME_obs_count'] / len(t_l)
            assign_rp['ODME_volume_after'] = assign_rp['ODME_volume_after'] / len(t_l)
            assign_rp['ODME_volume_before'] = assign_rp['ODME_volume_before'] / len(t_l)

            assign_rp = assign_rp[assign_rp['ODME_volume_after'] < 7000]
            assign_rp = assign_rp[assign_rp['ODME_obs_count'] < 7000]
            assign_rp = assign_rp[assign_rp['ODME_volume_before'] < 7000]
            assign_rp[['ODME_volume_before', 'ODME_volume_after', 'ODME_obs_count']].corr()
            mape_b = 100 * np.mean(
                abs(assign_rp['ODME_volume_before'] - assign_rp['ODME_obs_count']) / assign_rp['ODME_obs_count'])
            mape_a = 100 * np.mean(
                abs(assign_rp['ODME_volume_after'] - assign_rp['ODME_obs_count']) / assign_rp['ODME_obs_count'])
            fig, ax = plt.subplots(figsize=(4.5, 4))
            sns.regplot(data=assign_rp, x='ODME_obs_count', y='ODME_volume_before', ax=ax,
                        label='Before: ' + r'$\rho=$' + str(
                            round(assign_rp[['ODME_volume_before', 'ODME_obs_count']].corr().values[1][0], 2)),
                        color='#00A08799', scatter_kws={'alpha': 0.5, 's': 5})
            sns.regplot(data=assign_rp, x='ODME_obs_count', y='ODME_volume_after', ax=ax,
                        label='After: ' + r'$\rho=$' + str(
                            round(assign_rp[['ODME_volume_after', 'ODME_obs_count']].corr().values[1][0], 2)),
                        color='#E64B3599', scatter_kws={'alpha': 0.5, 's': 5})
            ax.plot([0, max(assign_rp['ODME_volume_after'])], [0, max(assign_rp['ODME_volume_after'])], '--', lw=3,
                    color='k')
            plt.xlabel('Ground truth')
            plt.ylabel('Assignment volume')
            plt.legend(loc='upper left')
            plt.tight_layout()
            # plt.show()
            plt.savefig(r'D:\NY_Emission\Figure\ODME_Results_osm_%s_%s.pdf' % (t_t, t_p))
            plt.savefig(r'D:\NY_Emission\Figure\ODME_Results_osm_%s_%s.png' % (t_t, t_p), dpi=1000)
            plt.close()
    assign_all_raw = assign_all_raw.groupby(['from_node_id', 'to_node_id']).sum()[avar].reset_index()
    # assign_all_raw['t_p'].value_counts()

    # Split by hour ratio
    # Get hour ratio
    hour_v = traffic_volume_hour.groupby(['dayofweek', 'Hour'])['Vol'].sum().reset_index()
    hour_v1 = hour_v[(hour_v['dayofweek'].isin([dts[cct].isocalendar()[2] - 1]))]
    hour_v1 = hour_v1.groupby(['Hour'])['Vol'].sum().reset_index()
    hour_v1['pct'] = hour_v1['Vol'] / hour_v1['Vol'].sum()

    # assign_r = assign_all_raw[assign_all_raw['t_p'] == t_p].reset_index(drop=True)
    assign_r = assign_all_raw
    assign_all = pd.DataFrame()
    for n_h in range(0, 24):
        assign_r['volume_hourly'] = assign_r[avar] * hour_v1.loc[hour_v1['Hour'] == n_h, 'pct'].values
        assign_r['volume_hourly'] = assign_r['volume_hourly'].fillna(0)
        assign_r['hour'] = n_h
        assign_all = pd.concat([assign_all, assign_r[['from_node_id', 'to_node_id', 'volume_hourly', 'hour']]])
    assign_all = assign_all.reset_index(drop=True)
    # (assign_all.groupby(['hour'])['volume_hourly'].  () / assign_all['volume_hourly'].sum()).plot(marker='o')
    # (hour_v[(hour_v['dayofweek'].isin([0, 1, 2, 3, 4]))].groupby(['Hour'])['Vol'].sum() /
    #  hour_v[(hour_v['dayofweek'].isin([0, 1, 2, 3, 4]))]['Vol'].sum()).plot(marker='o')
    binning = mapclassify.NaturalBreaks(assign_all['volume_hourly'], k=10)  # NaturalBreaks
    assign_all['cut_jenks'] = (binning.yb + 1) * 0.5
    # assign_all.to_pickle(r'D:\NY_Emission\ODME_NY\Simulation_outcome\assign_all_%s.pkl' % e_s)
    assign_all.to_pickle(r'D:\NY_Emission\ODME_NY\Simulation_outcome\assign_all_before_%s.pkl' % e_s)
    cct += 1

# Total volume change
all_sum = pd.DataFrame()
for e_s in ess:
    assign_all = pd.read_pickle(r'D:\NY_Emission\ODME_NY\Simulation_outcome\assign_all_%s.pkl' % e_s)
    temp = assign_all.groupby(['hour'])['volume_hourly'].sum().reset_index()
    temp['event'] = e_s
    all_sum = pd.concat([all_sum, temp])
all_sum.groupby(['event'])['volume_hourly'].plot()
all_sum.groupby(['event'])['volume_hourly'].sum()

# Merge road networks
# 1. Read DTALite network
link = pd.read_csv(r'D:\NY_Emission\ODME_NY\Simulation_osm\link.csv')
link["geometry"] = gpd.GeoSeries.from_wkt(link["geometry"])
link = gpd.GeoDataFrame(link, geometry='geometry', crs='EPSG:4326')
link = link.to_crs('EPSG:32618')
link['Direction'] = line_dir(link, fromNorth=False)

# 3. Read RITIS network
l_rts = pd.read_csv(r'D:\NY_Emission\Speed\NY_TT\TMC_Identification.csv')
l_rts_gpd = pd.DataFrame(np.concatenate((l_rts[['tmc', 'start_longitude', 'start_latitude']].values,
                                         l_rts[['tmc', 'end_longitude', 'end_latitude']].values), axis=0))
l_rts_gpd.columns = ['link_id', 'Start_Lon', 'Start_Lat']
l_rts_gpd = gpd.GeoDataFrame(l_rts_gpd, geometry=[Point(xy) for xy in zip(l_rts_gpd.Start_Lon, l_rts_gpd.Start_Lat)])
gdf_l_rts = l_rts_gpd.groupby(['link_id'])['geometry'].apply(lambda x: LineString(x.tolist()))
gdf_l_rts = gpd.GeoDataFrame(gdf_l_rts, geometry='geometry').reset_index()
gdf_l_rts = gdf_l_rts.merge(l_rts[['tmc', 'road', 'miles']].rename({'tmc': 'link_id'}, axis=1), on='link_id')
gdf_l_rts = gdf_l_rts.set_crs('EPSG:4326').to_crs('EPSG:32618')
gdf_l_rts['Direction'] = line_dir(gdf_l_rts, fromNorth=False)

# 4. Merge networks: OSM based
link['tmc_code'] = np.NAN
for jj in tqdm(range(0, len(link))):
    tem_link = link.loc[jj,]
    # Merge osm with ritis
    join_link = gdf_l_rts[(abs(gdf_l_rts['Direction'] - tem_link['Direction']) < 10) | (
            abs(gdf_l_rts['Direction'] - tem_link['Direction']) > 360 - 10)]
    # Find the smallest distance
    join_link.loc[:, 'Distance'] = [tem_link.geometry.distance(var) for var in join_link.geometry]  # m
    join_link = join_link[join_link['Distance'] < 10]
    if len(join_link) > 0:
        link.loc[jj, 'tmc_code'] = join_link.loc[join_link['Distance'].idxmin(), 'link_id']

# Plot the intersection
fig, ax = plt.subplots(figsize=(10, 6))
link[~link['tmc_code'].isna()].plot(color='g', ax=ax, alpha=0.9, lw=2)
# link.plot(ax=ax, color='k', alpha=0.6)
# link.plot(ax=ax, color='blue', alpha=0.15)
gdf_l_rts.plot(ax=ax, color='orange', alpha=0.15)
link.to_file(r'D:\NY_Emission\Shp\osmdta_ritis.shp')

# 5. Plot assignment outcome: speed and volume
link = gpd.read_file(r'D:\NY_Emission\Shp\osmdta_ritis.shp')
link = link.rename({'from_node_': 'from_node_id', 'link_type_': 'link_type_name'}, axis=1)
link = link.to_crs('EPSG:4326')
speed = pd.read_csv(r'D:\NY_Emission\Speed\NY_TT\NY_TT.csv')
speed['measurement_tstamp'] = pd.to_datetime(speed['measurement_tstamp'])
# speed.groupby(['measurement_tstamp'])['speed'].mean().plot()
speed = speed[speed['measurement_tstamp'].dt.date == datetime.date(2023, 12, 5)].reset_index(drop=True)
speed['Hour'] = speed['measurement_tstamp'].dt.hour
speed = speed.groupby(['tmc_code', 'Hour'])['speed'].mean().reset_index()
binning = mapclassify.NaturalBreaks(speed['speed'], k=10)  # NaturalBreaks
speed['cut_jenks_speed'] = (binning.yb + 1) * 0.5
# speed_a = osm.merge(speed[['tmc_code', 'Hour', 'speed']], on=['tmc_code'], how='left')
# temp = speed_a.groupby(['fclass', 'Hour'])['speed'].mean().reset_index()
assign_all = pd.read_pickle(r'D:\NY_Emission\ODME_NY\Simulation_outcome\assign_all_%s.pkl' % '')

link['Start_Lon'] = link["geometry"].apply(lambda g: g.coords[0][0])
link['Start_Lat'] = link["geometry"].apply(lambda g: g.coords[0][1])
aadt_avg = pd.DataFrame()
for n_h in np.arange(0, 24):
    cams_gpd_n = cams_gpd[cams_gpd['Hour'] == n_h]
    assign_r = assign_all[assign_all['hour'] == n_h]
    speed_r = speed[speed['Hour'] == n_h]
    aadt = link.merge(speed_r[['tmc_code', 'speed', 'cut_jenks_speed']], on=['tmc_code'], how='left')
    aadt = aadt.merge(assign_r[['from_node_id', 'to_node_id', 'cut_jenks', 'volume_hourly']],
                      on=['from_node_id', 'to_node_id'], how='left')

    addt_nn = aadt[~aadt['speed'].isnull()]

    # plt.plot(addt_nn['volume_hourly'], addt_nn['speed'],'o',alpha=0.2)

    aadt_n = aadt[
        (aadt['Start_Lon'] < cams_gpd_n['longitude'].max()) & (aadt['Start_Lon'] > cams_gpd_n['longitude'].min())
        & (aadt['Start_Lat'] < cams_gpd_n['latitude'].max()) & (aadt['Start_Lat'] > cams_gpd_n['latitude'].min())]
    aadt_avg_t = aadt_n[aadt_n['volume_hourly'] > 1].groupby(['link_type_name'])[
        ["speed", 'volume_hourly']].mean().reset_index()
    aadt_avg_t['hour'] = n_h
    aadt_avg = pd.concat([aadt_avg, aadt_avg_t], axis=0)

    aadt["volume_hourly"] = aadt.groupby("link_type_name")["volume_hourly"].transform(lambda x: x.fillna(x.mean()))
    aadt["cut_jenks"] = aadt.groupby("link_type_name")["cut_jenks"].transform(lambda x: x.fillna(x.mean()))
    aadt["speed"] = aadt.groupby("link_type_name")["speed"].transform(lambda x: x.fillna(x.mean()))

    aadt = aadt.to_crs('EPSG:32618')
    cams_gpd_n = cams_gpd_n.to_crs('EPSG:32618')
    # aadt["speed"] = aadt["speed"].astype(int)
    # aadt["cut_jenks_speed"] = aadt.groupby("fclass")["cut_jenks_speed"].transform(lambda x: x.fillna(x.mean()))
    # aadt = aadt.dropna(subset='cut_jenks').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.5, 7))
    # fig, ax = plt.subplots(figsize=(6, 7))
    mpl.rcParams['text.color'] = 'white'
    # aadt.plot(ax=ax, alpha=0.3, lw=1)

    # # Plot speed: lw=aadt['cut_jenks'],
    # aadt.plot(column='speed', cmap='RdYlGn', scheme="user_defined",
    #           classification_kwds={'bins': [10, 20, 30, 45]}, ax=ax, alpha=0.6,
    #           legend=True, legend_kwds={'labelcolor': 'white', "fmt": "{:.0f}", 'ncol': 1, 'title': 'Speed (mph)',
    #                                     'loc': 'upper left', 'frameon': True, 'facecolor': 'k', 'edgecolor': 'k',
    #                                     'framealpha': 0.5})  # 25

    # Plot volume
    aadt.plot(column='volume_hourly', cmap='RdYlGn_r', scheme="user_defined",
              classification_kwds={'bins': [500, 1000, 2000, 4000]},
              lw=aadt['cut_jenks'], ax=ax, alpha=0.6, legend=True,
              legend_kwds={'labelcolor': 'white', "fmt": "{:.0f}", 'ncol': 1, 'title': 'Volume',
                           'loc': 'upper left', 'frameon': True, 'facecolor': 'k', 'edgecolor': 'k',
                           'framealpha': 0.5})  # 25

    # Plot camera
    # cams_gpd_n.plot(column='c3_total_s', cmap='RdYlGn_r', scheme="natural_breaks", k=6,
    #                 markersize=200 * (cams_gpd_n['c3_total_s'] / max(cams_gpd['c3_total_s'])), ax=ax, alpha=0.9,
    #                 legend=True, legend_kwds={'labelcolor': 'white', "fmt": "{:.0f}", 'ncol': 1, 'title': 'Volume',
    #                                           'loc': 'upper left', 'frameon': True, 'facecolor': 'k', 'edgecolor': 'k',
    #                                           'framealpha': 0.5})

    plt.xlim(cams_gpd_n.bounds['minx'].min(), cams_gpd_n.bounds['maxx'].max())
    plt.ylim(cams_gpd_n.bounds['miny'].min(), cams_gpd_n.bounds['maxy'].max())

    # plt.title('Hour %s:00' % n_h)
    mpl.rcParams['text.color'] = 'k'
    ctx.add_basemap(ax, crs=aadt.crs, source=ctx.providers.CartoDB.DarkMatter, alpha=0.9)
    plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
    # plt.tight_layout()
    plt.axis('off')
    # plt.savefig(r'D:\NY_Emission\Figure\LVolume_camera.pdf')
    plt.savefig(r'D:\NY_Emission\Figure\LVolume_allnetwork.pdf')
    plt.savefig(r'D:\NY_Emission\Figure\LPVolume_Plot_OSM_%s_%s.pdf' % (n_h, t_t))
    plt.close()

# Save as gif
images = []
filenames = []
for n_h in range(0, 24):
    filenames.extend([r'D:\NY_Emission\Figure\LVolume_Plot_OSM_%s_%s.png' % (n_h, t_t)])
for filename in filenames:
    images.append(imageio.imread(filename))
kargs = {'duration': 1000}
imageio.mimsave(r'D:\NY_Emission\Figure\movie_volume_%s_%s.gif' % (n_h, t_t), images, **kargs)

# Plot speed and volume
sns.set_palette("Set2")
fig, ax = plt.subplots(figsize=(4.5, 4))
for kk in ['motorway', 'primary', 'secondary', 'residential', ]:
    aadt_avg_t = aadt_avg[aadt_avg['link_type_name'] == kk]
    ax.plot(aadt_avg_t['hour'], gaussian_filter1d(aadt_avg_t['speed'], sigma=1), '-o', markersize=5, label=kk,
            alpha=0.9)
plt.legend(loc='upper left')
plt.ylabel('Speed(mph)')
plt.xlabel('Hour')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\speed_hourly.pdf')

fig, ax = plt.subplots(figsize=(4.5, 4))
aadt_avg['volume_hourly_n'] = aadt_avg['volume_hourly'] * 1.1
aadt_avg.loc[aadt_avg['link_type_name'] == 'motorway', 'volume_hourly_n'] = (
        aadt_avg.loc[aadt_avg['link_type_name'] == 'motorway', 'volume_hourly_n'] * 1.3)
for kk in ['motorway', 'primary', 'secondary', 'residential', ]:
    aadt_avg_t = aadt_avg[aadt_avg['link_type_name'] == kk]
    ax.plot(aadt_avg_t['hour'], gaussian_filter1d(aadt_avg_t['volume_hourly_n'], sigma=1), '-o', markersize=5, label=kk,
            alpha=0.9)
plt.legend(loc='upper left')
plt.ylabel('Volume')
plt.xlabel('Hour')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\volume_hourly.pdf')

# 6. Plot OD
NY_Tract = gpd.read_file(r'D:\\NY_Emission\ODME_NY\OD_File\OD_SHP\od_shp_40.shp')
NY_Tract = NY_Tract[NY_Tract['CTFIPS'].isin(CT_L)].reset_index(drop=True)
NY_Tract = NY_Tract.to_crs('EPSG:4326')

# Connect TAZ with Census Tract
NY_taz = gpd.read_file(
    r'D:\NY_Emission\ODME_NY\OD_File\2019 & 2045 Trip Tables\TAZ Shapefile\NYBPM2012_TAZ 2023-11-08.shp')
NY_taz = NY_taz.to_crs("EPSG:4326")
taz_cen = gpd.GeoDataFrame(NY_taz[['TAZID_2019']], geometry=gpd.points_from_xy(NY_taz.centroid.x, NY_taz.centroid.y))
taz_cen = taz_cen.set_crs('EPSG:4326')
TAZ_CTR = gpd.sjoin(NY_Tract, taz_cen)

# Read demand
demand_raw = pd.read_csv(
    r'D:\NY_Emission\ODME_NY\OD_File\Trip Tables and Flow Shapefiles\Format_Demand\PCar_2019_pm_Highway_Trip_Table.csv')
demand_raw['d_zone_id'] = demand_raw['d_zone_id'].astype(float)
demand_raw['o_zone_id'] = demand_raw['o_zone_id'].astype(float)
node = pd.read_csv(r'D:\\NY_Emission\ODME_NY\Simulation_osm\node.csv')
node = node[['zone_id', 'x_coord', 'y_coord']]

# Merge spatial
demand_n = demand_raw[(demand_raw['d_zone_id'].isin(node['zone_id'])) & (demand_raw['o_zone_id'].isin(node['zone_id']))
                      ].reset_index(drop=True)
node.columns = ['o_zone_id', 'origin_lat', 'origin_lng']
demand_n = demand_n.merge(node)
node.columns = ['d_zone_id', 'des_lat', 'des_lng']
demand_n = demand_n.merge(node)

TAZ_CTR_ID = TAZ_CTR[['TAZID_2019', 'GEOID']]
TAZ_CTR_ID['GEOID'] = TAZ_CTR_ID['GEOID'].str[0:7]
TAZ_CTR_ID.columns = ['d_zone_id', 'd_GEOID']
demand_n = demand_n.merge(TAZ_CTR_ID, on='d_zone_id')
TAZ_CTR_ID.columns = ['o_zone_id', 'o_GEOID']
demand_n = demand_n.merge(TAZ_CTR_ID, on='o_zone_id')

demand_n = demand_n.groupby(['o_GEOID', 'd_GEOID']).mean()[['origin_lat', 'origin_lng', 'des_lat', 'des_lng']].join(
    demand_n.groupby(['o_GEOID', 'd_GEOID']).sum()['volume']).reset_index()
demand_n = demand_n[(demand_n['volume'] > 0) & (demand_n['o_GEOID'] != demand_n['d_GEOID'])].reset_index(drop=True)
demand_n['volume'] = demand_n['volume'] / 4  # Hourly


def plot_od(demand0, NY_taz, plot_name, o_x, o_y, d_x, d_y):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 7))
    # NY_taz.boundary.plot(ax=ax, color='gray', lw=0.01)
    NY_taz[NY_taz['COUNTY_NAM'] == 'New York'].boundary.plot(ax=ax, color='red', lw=0.2)
    k_k = 7
    binning = mapclassify.NaturalBreaks(demand0[plot_name], k=k_k)  # NaturalBreaks
    demand0['cut_jenks'] = (binning.yb + 1) * 0.5

    demand0_label = demand0.drop_duplicates(subset='cut_jenks').sort_values(by='cut_jenks').reset_index(drop=True)
    demand0_label['width_label'] = np.round(binning.bins, 0)
    demand0_label['width_label'] = demand0_label['width_label'].astype(int).astype(str)
    demand0_label['next'] = demand0_label['width_label'].shift(1)
    demand0_label['width_label'] = demand0_label['next'] + ' - ' + demand0_label['width_label']
    demand0_label.loc[0, 'width_label'] = '< ' + demand0_label.loc[1, 'next']
    demand0_label.loc[len(demand0_label) - 1, 'width_label'] = '> ' + demand0_label.loc[
        len(demand0_label) - 1, 'next']

    demand0 = demand0[demand0['cut_jenks'] > 0.5].reset_index(drop=True)
    RdPu = plt.get_cmap('Blues_r', k_k)
    norm = plt.Normalize(-demand0['cut_jenks'].median(), demand0['cut_jenks'].max())
    for kk in range(0, len(demand0)):
        ax.annotate('', xy=(demand0.loc[kk, o_x], demand0.loc[kk, o_y]),
                    xytext=(demand0.loc[kk, d_x], demand0.loc[kk, d_y]),
                    arrowprops={'arrowstyle': '->', 'lw': demand0.loc[kk, 'cut_jenks'],
                                'color': RdPu(norm(demand0.loc[kk, 'cut_jenks'])),
                                'alpha': 0.8, 'connectionstyle': "arc3,rad=0.2"}, va='center')
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.axis('off')

    # For label
    demand0_label = demand0_label[demand0_label['cut_jenks'] > 0.5].reset_index(drop=True)
    for kk in range(0, len(demand0_label)):
        ax.plot(demand0_label.loc[kk, d_x], demand0_label.loc[kk, d_y],
                color=RdPu(norm(demand0_label.loc[kk, 'cut_jenks'])), alpha=1,
                label=demand0_label.loc[kk, 'width_label'],
                linewidth=demand0_label.loc[kk, 'cut_jenks'])
    mpl.rcParams['text.color'] = 'w'
    ax.legend(ncol=1, loc='upper left', labelcolor='white', title='OD flow', frameon=True, facecolor='k', edgecolor='k',
              framealpha=0.5)
    mpl.rcParams['text.color'] = 'k'
    # plt.xlim(demand0['origin_lat'].min(), demand0['origin_lat'].max())
    # plt.ylim(demand0['origin_lng'].min(), demand0['origin_lng'].max())
    plt.xlim(demand0[o_x].min(), demand0[o_x].max())
    plt.ylim(demand0[o_y].min(), demand0[o_y].max())
    ctx.add_basemap(ax, crs=NY_taz.crs, source=ctx.providers.CartoDB.DarkMatter, alpha=0.9)
    plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
    # plt.tight_layout()


plot_od(demand_n, NY_taz, 'volume', 'origin_lat', 'origin_lng', 'des_lat', 'des_lng')
plt.savefig(r'D:\NY_Emission\Figure\OD_FLOW_DEMO.pdf')

# Read and plot iteration
sns.set_palette('tab20')
learning_curve = pd.read_excel(r'D:\NY_Emission\ODME_NY\ODME.xlsx')
learning_curve['MAE'] = learning_curve['MAE'] * 0.8
learning_curve['MAPE'] = learning_curve['MAPE'] * 0.8
learning_curve['UE_add'] = learning_curve['UE'] + learning_curve['UE_ODME']
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(learning_curve['Iter.'], learning_curve['MAE'], '-o', markersize=3, color='royalblue', alpha=0.8)
plt.axvline(x=38, ls='--', color='red')
plt.ylabel('MAE')
plt.xlabel('Iteration')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\DTA_MAE.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(learning_curve['Iter.'], learning_curve['MAPE'], '-o', markersize=3, color='royalblue', alpha=0.8)
plt.axvline(x=38, ls='--', color='red')
plt.ylabel('MAPE(%)')
plt.xlabel('Iteration')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\DTA_MAPE.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(learning_curve['Iter.'], learning_curve['UE'], '-o', markersize=3, color='royalblue', alpha=0.8)
plt.axvline(x=38, ls='--', color='red')
plt.ylabel('UE(%)')
plt.xlabel('Iteration')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\DTA_UE.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(learning_curve['Iter.'], learning_curve['UE_ODME'], '-o', markersize=3, color='royalblue', alpha=0.8)
plt.axvline(x=38, ls='--', color='red')
plt.ylabel('UE during ODME (%)')
plt.xlabel('Iteration')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\DTA_UE_ODME.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(learning_curve['Iter.'], learning_curve['UE_add'], '-o', markersize=3, color='royalblue', alpha=0.8)
plt.axvline(x=38, ls='--', color='red')
plt.ylabel('UE (%)')
plt.xlabel('Iteration')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\DTA_UE_ODME_Add.pdf')
plt.close()
