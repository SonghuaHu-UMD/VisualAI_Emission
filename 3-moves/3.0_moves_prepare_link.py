## Prepare MOVES at link level for all scenarios

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import geopandas as gpd
import requests
from scipy.spatial import Voronoi
import shapely
from shapely import geometry, ops
from pathlib import Path


def custom_round(x, base=5):
    return int(base * round(float(x) / base))


# Read signal network
osm_mt = pd.read_pickle(r'D:\NY_Emission\Shp\osmdta_ritis_signal.pkl')
osm_mt['linkID'] = osm_mt['from_node_'].astype(str) + '_' + osm_mt['to_node_id'].astype(str)
osm_mt = osm_mt.drop_duplicates(subset=['linkID']).reset_index(drop=True)

# Read vehicle reg
veh_typet = pd.read_csv(r'D:\NY_Emission\Volume_grth\Vehicle__Snowmobile__and_Boat_Registrations.csv')
# Read county
county_us = gpd.read_file(
    r'G:\Data\Dewey\SAFEGRAPH\Open Census Data\Census Website\2019\nhgis0011_shape\\US_county_2019.shp')
# Read speed
speeda = pd.read_csv(r'D:\NY_Emission\Speed\NY_TT\NY_TT.csv')
speeda = speeda[speeda['tmc_code'].isin(osm_mt['tmc_code'])].reset_index(drop=True)
speeda = speeda[['tmc_code', 'measurement_tstamp', 'speed']]
speeda['measurement_tstamp'] = pd.to_datetime(speeda['measurement_tstamp'])

# The scenarios we want to run
# cd: covid, te: thanksgiving; hf: henri flooding; ss: snow strom; '': normal
# ns: no signal; nsp: no speed; nvo: no volume; nv: no vehicle type; ncamera: no camera
# ncamera = nv + ns
ess = ['ncamera']
cct = 0
for e_ess in ess:
    # Read speed data
    # d_t = datetime.datetime(2023, 12, 5, 0, 0, 0)
    if e_ess == 'cd': d_t = datetime.datetime(2020, 3, 21)
    if e_ess == 'te': d_t = datetime.datetime(2021, 11, 24)
    if e_ess == 'hf': d_t = datetime.datetime(2021, 8, 22)
    if e_ess == 'ss': d_t = datetime.datetime(2022, 1, 29)
    # dt_s = d_t.strftime('%Y_%m_%d')

    speed = speeda[(speeda['measurement_tstamp'] < d_t + datetime.timedelta(hours=25)) & (
            speeda['measurement_tstamp'] >= d_t)].reset_index(drop=True)
    speed = (speed.set_index('measurement_tstamp').groupby(['tmc_code']).resample('60s').ffill().
             drop('tmc_code', axis=1).reset_index())

    # Read signal network
    osm_mt = pd.read_pickle(r'D:\NY_Emission\Shp\osmdta_ritis_signal.pkl')
    osm_mt['linkID'] = osm_mt['from_node_'].astype(str) + '_' + osm_mt['to_node_id'].astype(str)
    osm_mt = osm_mt.drop_duplicates(subset=['linkID']).reset_index(drop=True)
    osm_mt = osm_mt[['link_type_', 'linkID', 'tmc_code', 'Is_signal', 'geometry', 'Original', 'Cross']]

    # Assign each road segment with a speed profile: 60min*24
    ospeed = list(speed.drop_duplicates(subset=['measurement_tstamp'])['measurement_tstamp'])
    osm_mt = osm_mt.loc[osm_mt.index.repeat(60 * 25)].reset_index(drop=True)
    osm_mt['measurement_tstamp'] = ospeed * len(osm_mt.drop_duplicates(subset=['linkID']))
    osm_mt = osm_mt.merge(speed, on=['tmc_code', 'measurement_tstamp'], how='left')
    if e_ess == 'nsp':  # avg speed for each link type
        osm_mt['avg_speed'] = osm_mt.groupby(["link_type_"])["speed"].transform('mean')
        osm_mt["speed"] = osm_mt['avg_speed']
    else:
        osm_mt["speed"] = osm_mt.groupby(["link_type_", 'measurement_tstamp'])["speed"].transform(
            lambda x: x.fillna(x.mean()))
    osm_mt['hour'] = osm_mt['measurement_tstamp'].dt.hour
    # osm_mt.groupby(['hour'])['speed'].mean().plot()

    # Link profile
    osm_mt_link = osm_mt.drop_duplicates(subset=['linkID']).reset_index(drop=True)
    osm_mt_link = osm_mt_link.to_crs(epsg=3857)
    osm_mt_link['linkLength'] = osm_mt_link.geometry.length  # meter
    # gpd.GeoSeries(ops.linemerge(osm_mt_link.dissolve().geometry[0])).explode().to_file(r'D:\NY_Emission\Shp\temp1.shp')

    # Reset road type based on volume
    assign_all = pd.read_pickle(r'D:\NY_Emission\ODME_NY\Simulation_outcome\assign_all_.pkl')
    assign_all['linkID'] = assign_all['from_node_id'].astype(str) + '_' + assign_all['to_node_id'].astype(str)
    assign_all['volume_hourly'].describe()
    assign_all_lk = assign_all.groupby('linkID')['volume_hourly'].mean().reset_index()
    assign_all_lk = assign_all_lk.merge(osm_mt_link, on='linkID', how='right')
    # osm_mt.groupby(['Original']).count()

    # Generate second by second speed for each link
    for kk in tqdm(range(0, 24)):
        speed1 = osm_mt.loc[(osm_mt['measurement_tstamp'] <= d_t + datetime.timedelta(seconds=(kk + 1) * 3600)) & (
                osm_mt['measurement_tstamp'] >= d_t + datetime.timedelta(seconds=kk * 3600)),
        ['linkID', 'hour', 'measurement_tstamp', 'speed', 'Is_signal', 'Original', 'Cross']].reset_index(drop=True)
        speed1 = (
            speed1.groupby(['linkID']).apply(lambda x: x.set_index('measurement_tstamp').resample('s').ffill())).drop(
            ['linkID'], axis=1).reset_index()
        speed1 = speed1[speed1['hour'] == kk].reset_index(drop=True)
        speed1['grade'] = 0
        speed1['measurement_tstamp'] = (speed1['measurement_tstamp'] - d_t).dt.total_seconds()
        speed1 = speed1.drop(['hour'], axis=1)
        speed1.columns = ['linkID', 'secondID', 'speed', 'Is_signal', 'Original', 'Cross', 'grade']
        print(str(kk) + ': Length: ' + str(len(speed1)))

        # Generate path for the input path
        checkpoint_dir = Path(r'D:\NY_Emission\MOVES\input_%s' % e_ess)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if e_ess in ['ns', 'ncamera']:  # Output drivingCycle: no signal
            speed1[['linkID', 'secondID', 'speed', 'grade']].to_csv(
                r'D:\NY_Emission\MOVES\input_%s\drivingCycle_signal_%s.csv' % (e_ess, kk), index=False)
        else:  # Output drivingCycle: with signal:
            speed1['raw_speed'] = speed1['speed']
            speed1['speed'] = speed1['speed'] * speed1['Is_1']

            # Consider acc as 0.2g = 0.2*9.8*2.23694 mph/s = 4.385 mph/s
            speed1['cc_gp'] = (speed1['Is_1'] != speed1['Is_1'].shift()).cumsum()
            # speed1["consec"] = speed1.groupby('cc_gp').cumcount()
            speed1['ccount'] = (speed1['speed'] / 4.385).astype(int)  # how many second need to acc/dcc
            speed1['p_ctall'] = speed1.groupby(["cc_gp"])["linkID"].transform(
                "count")  # max second can be used to acc+dcc
            speed1['p_ctallmin'] = ((speed1['p_ctall'] - 2) // 2).astype(int)  # max second can be used to acc or dcc
            speed1['ccount'] = speed1[['p_ctallmin', 'ccount']].min(axis=1)

            slice_per_group = pd.Series(speed1.ccount.values, index=speed1.cc_gp).to_dict()
            idx_head5 = speed1.groupby('cc_gp', group_keys=False).apply(
                lambda g: g.head(slice_per_group.get(g.name))).index
            idx_tail5 = speed1.groupby('cc_gp', group_keys=False).apply(
                lambda g: g.tail(slice_per_group.get(g.name))).index
            idx_head1 = speed1.groupby('cc_gp', group_keys=False).head(1).index
            idx_tail1 = speed1.groupby('cc_gp', group_keys=False).tail(1).index

            # Remove those without signal
            indxn = speed1[(~speed1['Is_signal'])].index
            idx_head5 = idx_head5.difference(indxn)
            idx_tail5 = idx_tail5.difference(indxn)
            idx_head1 = idx_head1.difference(indxn)
            idx_tail1 = idx_tail1.difference(indxn)

            speed1['speed'] = speed1['speed'].fillna(method='ffill').fillna(method='bfill')
            speed1.loc[idx_head5, 'speed'] = np.nan
            speed1.loc[idx_tail5, 'speed'] = np.nan
            speed1['p_ct'] = speed1.groupby(["cc_gp"])["speed"].transform("count")
            speed1['speed'] = speed1['speed'] * (2 * speed1['p_ctall'] / (speed1['p_ct'] + speed1['p_ctall']))
            speed1.loc[idx_head1, 'speed'] = 0
            speed1.loc[idx_tail1, 'speed'] = 0
            speed1['speed'] = speed1['speed'].interpolate()

            # For those with too high speed: remove signal
            nosig = (list(set(speed1[(speed1['Original'] == 'residential') & (speed1['speed'] > 75)]['linkID'])) +
                     list(set(speed1[(speed1['Original'].isin(['secondary', 'primary'])) & (speed1['speed'] > 150)][
                                  'linkID'])))
            print(len(nosig))
            speed1.loc[speed1['linkID'].isin(nosig), 'speed'] = speed1.loc[speed1['linkID'].isin(nosig), 'raw_speed']

            print(str(kk) + ': Length: ' + str(len(speed1)))
            speed1[['linkID', 'secondID', 'speed', 'grade']].to_csv(
                r'D:\NY_Emission\MOVES\input_%s\drivingCycle_signal_%s.csv' % (e_ess, kk), index=False)


    # Generate link file (volume and speed)
    if e_ess in ['', 'ns', 'nsp', 'nvo', 'ncamera']:
        assign_all = pd.read_pickle(r'D:\NY_Emission\ODME_NY\Simulation_outcome\assign_all_.pkl')
    else:
        assign_all = pd.read_pickle(r'D:\NY_Emission\ODME_NY\Simulation_outcome\assign_all_%s.pkl' % e_ess)
    assign_all['linkID'] = assign_all['from_node_id'].astype(str) + '_' + assign_all['to_node_id'].astype(str)
    if e_ess in ['nvo']:  # avg volume for each link type
        assign_all = assign_all.merge(osm_mt_link[['linkID', 'link_type_']], on=['linkID'], how='left')
        assign_all['avg_volume'] = assign_all.groupby("link_type_")["volume_hourly"].transform('mean')
        # assign_all["avg_volume"] = assign_all.groupby("linkID")["volume_hourly"].transform(lambda x: x.fillna(x.mean()))
        assign_all['volume_hourly'] = assign_all["avg_volume"]
    for kk in tqdm(range(0, 24)):
        speedavg = osm_mt[osm_mt['hour'] == kk].reset_index(drop=True)
        speedavg = speedavg.groupby(['linkID'])['speed'].mean().reset_index()
        volumeavg = assign_all[assign_all['hour'] == kk].reset_index(drop=True)
        volumeavg = volumeavg.groupby(['linkID'])['volume_hourly'].mean().reset_index()
        speed2 = speedavg.merge(volumeavg, on=['linkID'], how='left')
        speed2 = speed2.merge(osm_mt_link[['linkID', 'linkLength']], on=['linkID'], how='left')
        speed2.columns = ['linkID', 'linkAvgSpeed', 'linkVolume', 'linkLength']
        speed2['countyID'] = 36061
        speed2['zoneID'] = 360610
        speed2['roadTypeID'] = 5
        speed2['linkDescription'] = 'Urban unrestricted'
        speed2['linkAvgGrade'] = 0
        speed2 = speed2.fillna(0)
        speed2[['linkID', 'countyID', 'zoneID', 'roadTypeID', 'linkLength', 'linkVolume', 'linkAvgSpeed',
                'linkDescription', 'linkAvgGrade']].to_csv(
            r'D:\NY_Emission\MOVES\input_%s\link_NYC_36061_%s.csv' % (e_ess, kk), index=False)

    # Generate source type
    '''
    Motorcycles: 11; Passenger Cars: 21; Passenger Trucks: 31; Light Commercial Trucks: 32; Other Buses: 41
    Transit Buses: 42; School Buses: 43; Refuse Trucks: 51; Short-Haul Single Unit Trucks: 52
    Long-Haul Single Unit Trucks: 53; Motor Homes: 54; Short-Haul Combination Trucks: 61; Long-Haul Combination Trucks: 62
    '''
    # Get car type distribution from camera
    count_df = pd.read_pickle(r'D:\NY_Emission\Video_Process\count_df_new_final.pkl')
    p_c = ['c3_car_s', 'c3_ptruck_s', 'c3_htruck_s', 'c3_ltruck_s', 'c3_sbus_s', 'c3_bus_s', 'c3_mcycle_s']
    ct_weight = count_df.groupby(['file', 'Hour'])['frame'].count() / 3600
    ct_each = count_df.groupby(['file', 'Hour'])[p_c].sum().div(ct_weight, axis=0).reset_index()
    ct_each['id'] = ct_each['file'].str.replace('.mp4', '')
    ct_each['sum'] = ct_each[p_c].sum(axis=1)

    avg_ratio = pd.DataFrame(ct_each[p_c].sum() / ct_each['sum'].sum()).reset_index()
    avg_ratio.columns = ['sourceTypeID', 'ratio']
    avg_ratio = avg_ratio.replace(
        {'sourceTypeID': {'c3_car_s': 21, 'c3_ptruck_s': 31, 'c3_htruck_s': 61, 'c3_ltruck_s': 52,
                          'c3_sbus_s': 43, 'c3_bus_s': 42, 'c3_mcycle_s': 11, }})
    avg_ratio = pd.concat([avg_ratio, pd.DataFrame({'sourceTypeID': ['32', '41', '51', '53', '54', '62'],
                                                    'ratio': [0, 0, 0, 0, 0, 0]})], axis=0)
    avg_ratio['sourceTypeID'] = avg_ratio['sourceTypeID'].astype(int)

    for kk in p_c:
        ct_each[kk] = ct_each[kk] / ct_each['sum']
    for kk in ['32', '41', '51', '53', '54', '62']:
        ct_each[kk] = 0
    ct_each = pd.melt(ct_each, id_vars=['id', 'Hour'], value_vars=p_c + ['32', '41', '51', '53', '54', '62'])
    ct_each = ct_each.replace({'variable': {'c3_car_s': 21, 'c3_ptruck_s': 31, 'c3_htruck_s': 61, 'c3_ltruck_s': 52,
                                            'c3_sbus_s': 43, 'c3_bus_s': 42, 'c3_mcycle_s': 11, }})
    ct_each['variable'] = ct_each['variable'].astype(int)
    ct_each = ct_each.sort_values(by=['id', 'Hour', 'variable'], ascending=True).reset_index(drop=True)

    # From camera: Generate a Thiessen Polygon for node-link matching
    cams = pd.DataFrame(requests.get(r'https://webcams.nyctmc.org/api/cameras/').json())
    url_list = list(cams['imageUrl'])
    cams_gpd = gpd.GeoDataFrame(cams, geometry=gpd.points_from_xy(x=cams.longitude, y=cams.latitude))
    cams_gpd = cams_gpd.set_crs('EPSG:4326')
    cams_gpd = cams_gpd[cams_gpd['id'].isin(ct_each['id'])].reset_index(drop=True)

    x = cams_gpd.geometry.x.values
    y = cams_gpd.geometry.y.values
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

    # Sjoin with camera
    SInBG = gpd.sjoin(cams_gpd, voronois, how='inner', op='within').reset_index(drop=True)
    SInBG_index = SInBG[['id', 'index']].drop_duplicates(subset='index', keep='first')
    voronois = voronois.merge(SInBG_index, on='index', how='left')

    # Sjoin with road link
    voronois = voronois.to_crs('EPSG:3857')
    SInBG = gpd.sjoin(osm_mt_link, voronois, how='left', predicate='intersects').reset_index(drop=True)
    SInBG = SInBG[['id', 'linkID']].drop_duplicates(subset='linkID', keep='first')


    for kk in range(0, 24):
        ct_each1 = ct_each[ct_each['Hour'] == kk].sort_values(by=['id', 'variable']).reset_index(drop=True)
        ct_each1 = ct_each1[['id', 'variable', 'value']]
        ct_each1 = ct_each1.merge(SInBG, on='id', how='right')
        ct_each1 = ct_each1.pivot(index='linkID', columns='variable', values='value')
        ct_each1 = ct_each1.loc[:, pd.notnull(ct_each1.columns)]
        ct_each1 = ct_each1.unstack().reset_index().sort_values(by=['linkID', 'variable']).reset_index(drop=True)
        ct_each1.columns = ['sourceTypeID', 'linkID', 'sourceTypeHourFraction']
        ct_each1["sourceTypeHourFraction"] = ct_each1.groupby("sourceTypeID")["sourceTypeHourFraction"].transform(
            lambda x: x.fillna(x.mean()))
        print(len(ct_each1) / 13)
        # ct_each1['sourceTypeHourFraction'].isnull().sum()
        ct_each1.to_csv(r'D:\NY_Emission\MOVES\input_%s\sourceTypeDistribution_NYC_%s.csv' % (e_ess, kk), index=False)

        # Avg output: for non-vehicle case
        if e_ess in ['nv', 'ncamera']:
            ct_each2 = ct_each1.copy()
            ct_each2 = ct_each2.merge(avg_ratio, on='sourceTypeID')
            ct_each2 = ct_each2.sort_values(by=['linkID', 'sourceTypeID']).reset_index(drop=True)
            ct_each2['sourceTypeHourFraction'] = ct_each2['ratio']
            ct_each2.drop('ratio', axis=1, inplace=True)
            ct_each2.to_csv(r'D:\NY_Emission\MOVES\input_nv\sourceTypeDistribution_NYC_%s.csv' % kk, index=False)

    # Vehicle age distribution
    veh_typet = veh_typet[veh_typet['Record Type'].isin(['VEH', 'TRL'])].reset_index(drop=True)
    veh_typet['sourceType'] = 21
    veh_typet.loc[veh_typet['Body Type'].isin(['SUBN', 'PICK', 'VAN']), 'sourceType'] = 31
    veh_typet.loc[veh_typet['Body Type'].isin(['4DSD', '2DSD', 'CONV', 'ATV', 'SEDN', 'TAXI']), 'sourceType'] = 21
    veh_typet.loc[veh_typet['Body Type'].isin(['BUS']), 'sourceType'] = 42
    veh_typet.loc[veh_typet['Body Type'].isin(['MCY']), 'sourceType'] = 11
    veh_typet.loc[veh_typet['Body Type'].isin(['H/WH', 'H/TR']), 'sourceType'] = 54
    veh_typet.loc[veh_typet['Body Type'].isin(['LTRL', 'UTIL', 'DELV']), 'sourceType'] = 52
    veh_typet.loc[veh_typet['Body Type'].isin(['SEMI', 'TRLR', 'DUMP', 'TRACTOR', 'FLAT', 'P/SH']), 'sourceType'] = 61
    veh_typet['age'] = 2024 - veh_typet['Model Year']
    veh_typet = veh_typet[(veh_typet['age'] <= 50) & (veh_typet['age'] >= 0)]
    veh_dis = (veh_typet['Body Type'].value_counts() / len(veh_typet)).reset_index()
    veh_age_dis = (veh_typet.groupby('sourceType')['age'].value_counts() / veh_typet.groupby('sourceType')[
        'age'].count()).reset_index()
    veh_age_dis.columns = ['sourceTypeID', 'ageID', 'ageFraction']
    veh_age_dis['yearID'] = 2023

    stl = [11, 21, 31, 32, 41, 42, 43, 51, 52, 53, 54, 61, 62]
    veh_age_all = pd.DataFrame({'sourceTypeID': np.repeat(stl, 51), "ageID": list(range(0, 51)) * len(stl)})
    veh_age_all = veh_age_all.merge(veh_age_dis, on=['sourceTypeID', "ageID"], how='left')
    veh_age_all["ageFraction"] = veh_age_all.groupby(["ageID"])['ageFraction'].transform(
        lambda x: x.fillna(x.mean()))
    veh_age_all['yearID'] = 2023
    veh_age_all.to_csv(r'D:\NY_Emission\MOVES\input_%s\ageDistribution_2023.csv' % e_ess, index=False)

    # Meteorology for new york
    weatherr = pd.read_excel(r'D:\NY_Emission\MOVES\Weather.xlsx')
    weatherr['Date'] = pd.to_datetime(weatherr['Date'])
    weatherr['Temperature'] = weatherr.Temperature.str.extract('(\d+)')
    weatherr['Temperature'] = weatherr['Temperature'].astype(int).apply(lambda x: custom_round(x, base=5))
    weatherr['Humidity'] = weatherr.Humidity.str.extract('(\d+)')
    weatherr['Humidity'] = weatherr['Humidity'].astype(int).apply(lambda x: custom_round(x, base=5))

    weather = pd.DataFrame(
        {'monthID': [d_t.month] * 24, 'zoneID': [36061] * 24, "hourID": range(0, 24),
         "temperature": weatherr.loc[weatherr['Date'] == d_t, 'Temperature'].to_list(),
         "realHumidity": weatherr.loc[weatherr['Date'] == d_t, 'Humidity'].to_list()})

    weather.to_csv(r'D:\NY_Emission\MOVES\input_%s\meteorology_NYC_36061_01.csv' % e_ess, index=False)

    # Generate batchmode
    batchmode = pd.DataFrame()
    batchmode['taskID'] = range(0, 24)
    batchmode['region'] = 'newyork'
    batchmode['calendarYear'] = 2023
    batchmode['meteorologyFileName'] = 'meteorology_NYC_36061_01.csv'
    batchmode['sourceTypeDistributionFileName'] = ['sourceTypeDistribution_NYC_%s.csv' % ii for ii in range(0, 24)]
    batchmode['ageDistributionFileName'] = 'ageDistribution_2023.csv'
    batchmode['linkFileName'] = ['link_NYC_36061_%s.csv' % ii for ii in range(0, 24)]
    batchmode['driveSchedule/OpModeDistributionFileName'] = ['drivingCycle_signal_%s.csv' % ii for ii in range(0, 24)]
    batchmode['opmode(o)/cycle(d)/speed(v)'] = 'd'
    batchmode.to_csv(r'D:\NY_Emission\MOVES\input_%s\batchmode.csv' % e_ess, index=False)

    cct += 1
