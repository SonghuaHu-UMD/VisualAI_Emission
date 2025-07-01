# Do the DTA and output the link volume for each period under different scenarios
from sys import stdin

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import subprocess
import os
import osm2gmns as og

pd.options.mode.chained_assignment = None
os.chdir(r'D:\NY_Emission\ODME_NY\Simulation_osm')


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
t_t = 'Car'

### 1. Get simulation area (Census Tract)
NY_Tract = gpd.read_file(r'D:\NY_Emission\ODME_NY\OD_File\OD_SHP\od_shp_40.shp')
NY_Tract = NY_Tract[NY_Tract['CTFIPS'].isin(CT_L)].reset_index(drop=True)
NY_Tract = NY_Tract.to_crs('EPSG:4326')
# NY_Tract.plot()

# Connect TAZ with Census Tract
NY_taz = gpd.read_file(
    r'D:\NY_Emission\ODME_NY\OD_File\2019 & 2045 Trip Tables\TAZ Shapefile\NYBPM2012_TAZ 2023-11-08.shp')
NY_taz = NY_taz.to_crs("EPSG:4326")
taz_cen = gpd.GeoDataFrame(NY_taz[['TAZID_2019']], geometry=gpd.points_from_xy(NY_taz.centroid.x, NY_taz.centroid.y))
taz_cen = taz_cen.set_crs('EPSG:4326')
TAZ_CTR = gpd.sjoin(NY_Tract, taz_cen)
TAZ_CTR_ID = TAZ_CTR[['TAZID_2019', 'GEOID']]
Need_TAZ = NY_taz[NY_taz['TAZID_2019'].isin(set(TAZ_CTR['TAZID_2019']))].reset_index(drop=True)
# Need_TAZ.plot()

### 3. Prepare for DTALite
# Read ground truth from DOT
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
traffic_v_mean = traffic_volume_hour.groupby(['SegmentID', 'dayofweek', 'Hour'])['Vol'].median().reset_index()
traffic_v_meana = traffic_v_mean[traffic_v_mean['dayofweek'] == 1]
# traffic_v_mean.groupby(['Hour'])['Vol'].mean().plot(marker='o')

# Get time series ratio
all_visits = pd.read_pickle(r'D:\NY_Emission\ODME_NY\OD_File\MDLD\Timeseries.pkl')
all_visits['Timestamp'] = pd.to_datetime(all_visits['Timestamp'])
all_visits['dayofweek'] = all_visits['Timestamp'].dt.dayofweek
all_visits['hour'] = all_visits['Timestamp'].dt.hour

# Get demand changes under abnormal period
ABN_CAL = 'cg'  # 'cg'
event_ratio = pd.read_pickle(r'D:\\NY_Emission\ODME_NY\OD_File\MDLD\od_event_ratio.pkl')
cg_ratio = pd.read_pickle(r'D:\\NY_Emission\ODME_NY\OD_File\MDLD\od_cg_ratio.pkl')
mvolume = pd.read_pickle(r'D:\NY_Emission\CongestionPricing\volume_change.pkl')

'''
# # Build link from osm
# net_osm = og.getNetFromFile(r'D:\\NY_Emission\Shp\OSM\planet_-74.461_40.495_8139facd.osm.pbf', default_lanes=True,
#                             default_speed=True, default_capacity=True)
# og.outputNetToCSV(net_osm, output_folder=r'D:\\NY_Emission\ODME_NY\Simulation_osm\network_osm')

# Merge sensor (lion) with DTALite road network
# Read lion network
lion = gpd.read_file(r'D:\\NY_Emission\Shp\AADT_New\Dynamic_count.shp')
lion = lion.drop_duplicates(['SegmentID']).reset_index(drop=True)
lion = lion.to_crs('EPSG:32618')
lion['Direction'] = line_dir(lion, fromNorth=False)

# Read osm network
link = pd.read_csv(r'D:\\NY_Emission\ODME_NY\Simulation_osm\network_osm\link.csv')
link["geometry"] = gpd.GeoSeries.from_wkt(link["geometry"])
link = gpd.GeoDataFrame(link, geometry='geometry', crs='EPSG:4326')
link['Direction'] = line_dir(link, fromNorth=False)
link = link.to_crs('EPSG:32618')

# Merge two network
lion['from_node_id'] = np.NAN
lion['to_node_id'] = np.NAN
for jj in tqdm(range(0, len(lion))):
    tem_link = lion.loc[jj,]
    join_link = link[(abs(link['Direction'] - tem_link['Direction']) < 10) | (
            abs(link['Direction'] - tem_link['Direction']) > 360 - 10)]
    # Find the smallest distance
    join_link.loc[:, 'Distance'] = [tem_link.geometry.distance(var) for var in join_link.geometry]  # m
    join_link = join_link[join_link['Distance'] < 10]
    if len(join_link) > 0:
        lion.loc[jj, 'from_node_id'] = join_link.loc[join_link['Distance'].idxmin(), 'from_node_id']
        lion.loc[jj, 'to_node_id'] = join_link.loc[join_link['Distance'].idxmin(), 'to_node_id']
lion.to_file(r'D:\\NY_Emission\ODME_NY\OD_File\OD_SHP\lion_osm.shp')
'''

# Prepare all files for DTALite and Run the simulation
lion = gpd.read_file(r'D:\NY_Emission\ODME_NY\OD_File\OD_SHP\lion_osm.shp')
all_demand_Des = []
for t_p in ['am', 'pm', 'md', 'nt1', 'nt2']:  #
    if t_p == 'am':
        t_l = [6, 7, 8, 9]
        time_period = '0600_0900'
        peak_time = '0830'
    elif t_p == 'md':
        t_l = [10, 11, 12, 13, 14]
        time_period = '1000_1400'
        peak_time = '1000'
    elif t_p == 'pm':
        t_l = [15, 16, 17, 18]
        time_period = '1500_1800'
        peak_time = '1730'
    elif t_p == 'nt1':
        t_l = [19, 20, 21, 22, 23]
        time_period = '1900_2300'
        peak_time = '1900'
    elif t_p == 'nt2':
        t_l = [0, 1, 2, 3, 4, 5]
        time_period = '0000_0500'
        peak_time = '0500'

    # Read node
    node = pd.read_csv(r'D:\NY_Emission\ODME_NY\Simulation_osm\network_osm\node.csv')
    node = gpd.GeoDataFrame(node, geometry=gpd.points_from_xy(node.x_coord, node.y_coord), crs="EPSG:4326")

    # Read link
    link = pd.read_csv(r'D:\NY_Emission\ODME_NY\Simulation_osm\network_osm\link.csv')
    link["geometry"] = gpd.GeoSeries.from_wkt(link["geometry"])
    # link['capacity'] = link['capacity'] * len(t_l)
    link = gpd.GeoDataFrame(link, geometry='geometry', crs='EPSG:4326')

    # Check link and node: all link's node should be found in node.csv
    link_node = set(list(set(link['from_node_id'])) + list(set(link['to_node_id'])))
    node_node = set(node['node_id'])
    print(len(link_node & node_node) / len(link_node))
    node = node[node['node_id'].isin(link_node)].reset_index(drop=True)

    # node and TAZ join: assign zone id (TAZ) to node
    SInBG = gpd.sjoin(node, Need_TAZ, how='inner', op='within').reset_index(drop=True)
    SInBG_index = SInBG[['node_id', 'TAZID_2019']]
    node_speed = link.groupby('from_node_id')[['free_speed', 'capacity']].mean().reset_index()
    node_speed.columns = ['node_id', 'node_speed', 'node_cap']
    SInBG_index = SInBG_index.merge(node_speed, on='node_id', how='left')
    idx = SInBG_index.groupby(['TAZID_2019'])['node_speed'].transform(max) == SInBG_index['node_speed']
    SInBG_index1 = SInBG_index[idx].reset_index(drop=True)
    idx = SInBG_index1.groupby(['TAZID_2019'])['node_cap'].transform(max) == SInBG_index1['node_cap']
    SInBG_index2 = SInBG_index1[idx]
    SInBG_indexf = SInBG_index2.groupby('TAZID_2019').sample(n=1)[['node_id', 'TAZID_2019']]
    node = node.merge(SInBG_indexf, on='node_id', how='left')
    node['zone_id'] = node['TAZID_2019']
    node = node.drop('TAZID_2019', axis=1)

    # We don't need such a large network
    Need_TAZ_D = Need_TAZ.dissolve()
    Need_TAZ_D['geometry'] = Need_TAZ_D['geometry'].convex_hull
    SInBG = gpd.sjoin(node, Need_TAZ_D, how='inner', op='within').reset_index(drop=True)
    node = node[node['node_id'].isin(SInBG['node_id'])].reset_index(drop=True)
    link = link[(link['from_node_id'].isin(SInBG['node_id'])) & (link['to_node_id'].isin(SInBG['node_id']))
                ].reset_index(drop=True)

    # link type
    link = link[link['link_type_name'].isin(['residential', 'secondary', 'primary', 'tertiary', 'motorway_link',
                                             'motorway', 'unclassified', 'trunk'])].reset_index(drop=True)
    link_type = link.groupby(['link_type', 'link_type_name'])[['free_speed', 'capacity']].mean().reset_index()

    # Output
    node.to_csv(r'D:\NY_Emission\ODME_NY\Simulation_osm\node.csv', index=0)
    link.to_csv(r'D:\NY_Emission\ODME_NY\Simulation_osm\link.csv', index=0)

    # Select demand we need
    demand = pd.read_csv(r'D:\NY_Emission\ODME_NY\OD_File\Format_Demand\P%s_%s_Trip_Table.csv' % (t_t, t_p))
    demand['d_zone_id'] = demand['d_zone_id'].astype(float)
    demand['o_zone_id'] = demand['o_zone_id'].astype(float)
    demand_need = demand[(demand['d_zone_id'].isin(node['zone_id'])) &
                         (demand['o_zone_id'].isin(node['zone_id']))].reset_index(drop=True)
    demand_need = demand_need[demand_need['o_zone_id'] != demand_need['d_zone_id']].reset_index(drop=True)
    demand_need = demand_need[demand_need['volume'] > 1]
    all_demand_Des.append([t_p, len(demand_need), sum(demand_need['volume']), len(set(demand_need['o_zone_id'])),
                           len(set(demand_need['d_zone_id'])), max(demand_need['volume']), min(demand_need['volume']),
                           np.std(demand_need['volume'])])

    event_ratio_need = event_ratio[event_ratio['hour'].isin(t_l)].reset_index(drop=True)
    event_ratio_need = event_ratio_need.groupby(['start_admin2', 'end_admin2']).sum().reset_index()

    cg_ratio_need = cg_ratio[cg_ratio['hour'].isin(t_l)].reset_index(drop=True)
    cg_ratio_need = cg_ratio_need.groupby(['start_admin2', 'end_admin2']).sum().reset_index()

    if ABN_CAL == 'event':
        l_d_d = ['ss', 'hf', 'te', 'cd']
    elif ABN_CAL == 'cg':
        l_d_d = ['cg_2', 'cg_4', 'cg_6', 'cg_8']
    else:
        l_d_d = ['']
    for d_d in l_d_d:
        if ABN_CAL == 'event':
            TAZ_CTR_ID = TAZ_CTR[['TAZID_2019', 'GEOID']]
            TAZ_CTR_ID['CTFIPS'] = TAZ_CTR_ID['GEOID'].str[0:5]
            TAZ_CTR_ID = TAZ_CTR_ID[['CTFIPS', 'TAZID_2019']]
            TAZ_CTR_ID.columns = ['start_admin2', 'o_zone_id']
            demand_need_abn = demand_need.merge(TAZ_CTR_ID, on='o_zone_id')
            TAZ_CTR_ID.columns = ['end_admin2', 'd_zone_id']
            demand_need_abn = demand_need_abn.merge(TAZ_CTR_ID, on='d_zone_id')

            print(sum((event_ratio_need['total_trips_%s' % d_d] - event_ratio_need['total_trips_avg'])) / sum(
                event_ratio_need['total_trips_avg']))
            event_ratio_need['changes'] = event_ratio_need['total_trips_%s' % d_d] / event_ratio_need['total_trips_avg']
            event_ratio_need1 = event_ratio_need[['start_admin2', 'end_admin2', 'changes']]
            demand_need_abn = demand_need_abn.merge(event_ratio_need1, on=['start_admin2', 'end_admin2'])
            demand_need_abn['volume'] = demand_need_abn['volume'] * demand_need_abn['changes']
            demand_need_abn = demand_need_abn[demand_need_abn['volume'] > 0.1].reset_index(drop=True)
            demand_need_abn[['d_zone_id', 'o_zone_id', 'volume']].to_csv(
                r'D:\NY_Emission\ODME_NY\Simulation_osm\demand.csv', index=0)
        elif ABN_CAL == 'cg':
            TAZ_CTR_ID = TAZ_CTR[['TAZID_2019', 'GEOID']]
            TAZ_CTR_ID['CTFIPS'] = TAZ_CTR_ID['GEOID'].str[0:5]
            TAZ_CTR_ID = TAZ_CTR_ID[['CTFIPS', 'TAZID_2019']]
            TAZ_CTR_ID.columns = ['start_admin2', 'o_zone_id']
            demand_need_abn = demand_need.merge(TAZ_CTR_ID, on='o_zone_id')
            TAZ_CTR_ID.columns = ['end_admin2', 'd_zone_id']
            demand_need_abn = demand_need_abn.merge(TAZ_CTR_ID, on='d_zone_id')

            rt_need = mvolume.loc[
                (mvolume['week'] == int(d_d.split('_')[1])) & (mvolume['cartype'] == 'car'), 'pct_change'].item()
            rt_now = sum((cg_ratio_need['total_trips_%s' % d_d])) / sum(cg_ratio_need['total_trips_avg'])
            cg_ratio_need['total_trips_%s' % d_d] = cg_ratio_need['total_trips_%s' % d_d] * (
                    (1 + rt_need / 100) / rt_now)

            cg_ratio_need['changes'] = cg_ratio_need['total_trips_%s' % d_d] / cg_ratio_need['total_trips_avg']
            cg_ratio_need1 = cg_ratio_need[['start_admin2', 'end_admin2', 'changes']]
            demand_need_abn = demand_need_abn.merge(cg_ratio_need1, on=['start_admin2', 'end_admin2'])
            demand_need_abn['volume'] = demand_need_abn['volume'] * demand_need_abn['changes']
            demand_need_abn = demand_need_abn[demand_need_abn['volume'] > 0.1].reset_index(drop=True)
            demand_need_abn[['d_zone_id', 'o_zone_id', 'volume']].to_csv(
                r'D:\NY_Emission\ODME_NY\Simulation_osm\demand.csv', index=0)
        else:
            demand_need = demand_need[demand_need['volume'] > 0.1].reset_index(drop=True)
            demand_need[['d_zone_id', 'o_zone_id', 'volume']].to_csv(
                r'D:\NY_Emission\ODME_NY\Simulation_osm\demand.csv', index=0)

        # Generate sensor data for ODME
        t_mean = traffic_v_meana[traffic_v_meana['Hour'].isin(t_l)].groupby(['SegmentID'])['Vol'].sum().reset_index()
        lion_label = lion[['from_node_', 'to_node_id', 'SegmentID']].dropna().reset_index(drop=True)
        lion_label = lion_label.merge(t_mean, on='SegmentID').reset_index()
        lion_label = lion_label[['index', 'from_node_', 'to_node_id', 'Vol']]
        lion_label.columns = ['sensor_id', 'from_node_id', 'to_node_id', 'count']
        lion_label = lion_label[(lion_label['from_node_id'].isin(SInBG['node_id'])) & (
            lion_label['to_node_id'].isin(SInBG['node_id']))].reset_index(drop=True)
        lion_label['scenario_index'] = 0
        lion_label['activate'] = 1
        lion_label['demand_period'] = t_p
        if ABN_CAL == 'event':
            lion_label['count'] = lion_label['count'] * (
                    sum(event_ratio_need['total_trips_%s' % d_d]) / sum(event_ratio_need['total_trips_avg']))
            print(sum(event_ratio_need['total_trips_%s' % d_d]) / sum(event_ratio_need['total_trips_avg']))
        elif ABN_CAL == 'cg':
            lion_label['count'] = lion_label['count'] * (
                    sum(cg_ratio_need['total_trips_%s' % d_d]) / sum(cg_ratio_need['total_trips_avg']))
            print(sum(cg_ratio_need['total_trips_%s' % d_d]) / sum(cg_ratio_need['total_trips_avg']))
        lion_label.to_csv(r'D:\NY_Emission\ODME_NY\Simulation_osm\sensor_data.csv', index=False)

        # Generate demand period
        demand_period = pd.DataFrame(
            {'first_column': [0], "demand_period_id": 1, "demand_period": t_p, "notes": 'weekday',
             "time_period": time_period, "peak_time": peak_time})
        demand_period.to_csv(r'D:\NY_Emission\ODME_NY\Simulation_osm\demand_period.csv', index=False)
        demand_file_list = pd.DataFrame(
            {'first_column': [0], "file_sequence_no": 1, "scenario_index_vector": 0, "file_name": "demand.csv",
             "demand_period": t_p, "mode_type": 'auto', "format_type": "column", "scale_factor": 1,
             "departure_time_profile_no": 1})
        demand_file_list.to_csv(r'D:\NY_Emission\ODME_NY\Simulation_osm\demand_file_list.csv', index=False)

        # Generate period ratio
        hour_v = all_visits.groupby(['dayofweek', 'hour'])['Visits'].sum().reset_index()
        hour_v1 = hour_v[(hour_v['dayofweek'] == 1)]
        hour_v1['pct'] = hour_v1['Visits'] / hour_v1['Visits'].sum()
        hour_v1 = hour_v1.loc[hour_v1.index.repeat(60 / 5)]
        hour_v1['pct'] = hour_v1['pct'] / (60 / 5)
        hour_v1['second'] = range(0, 1440, 5)
        hour_v1['second'] = 'T' + hour_v1['second'].astype(str).str.zfill(4)
        hour_v2 = hour_v1[['pct']].T
        hour_v2.columns = hour_v1['second'].tolist()
        hour_v2['first_column'] = 0
        hour_v2['departure_time_profile_no'] = 1
        hour_v2['time_period'] = time_period
        hour_v2[['first_column', 'departure_time_profile_no', 'time_period'] + hour_v1['second'].tolist()].to_csv(
            r'D:\NY_Emission\ODME_NY\Simulation_osm\departure_time_profile.csv', index=False)

        # Run assignment
        subprocess.call([r"D:\NY_Emission\ODME_NY\Simulation_osm\DTALite_230915.exe"])

        # Read and save results
        assign_r = pd.read_csv(r'D:\NY_Emission\ODME_NY\Simulation_osm\link_performance_s0_25nb.csv')
        assign_r.to_csv(
            r'D:\NY_Emission\ODME_NY\Simulation_outcome\osm_link_performance_%s_%s_%s.csv' % (t_t, t_p, d_d))

