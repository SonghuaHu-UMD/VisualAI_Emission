## Prepare MOVES at link level for all scenarios

import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import geopandas as gpd
import requests
from scipy.spatial import Voronoi
import shapely
from pathlib import Path
from utils import generate_signal_phase


def custom_round(x, base=5):
    return int(base * round(float(x) / base))


def bpr_function(x, ffs, alpha=0.15, beta=4):
    """BPR speed-flow function. Default params per HCM: alpha=0.15, beta=4."""
    return ffs / (1 + alpha * np.power(x, beta))


def _redistribute_peak(assign_all, peak_hours, before_hour, after_hour, r_rr):
    """Redistribute volume from peak hours to adjacent shoulder hours."""
    assign_all.loc[assign_all['hour'].isin(peak_hours), 'volume_new'] = (
            assign_all.loc[assign_all['hour'].isin(peak_hours), 'volume_new'] * (1 - r_rr / 2))
    sum_volume = sum(assign_all.loc[assign_all['hour'].isin(peak_hours), 'volume_new'] * (r_rr / 2))
    for shoulder_hour in [before_hour, after_hour]:
        mask = assign_all['hour'] == shoulder_hour
        assign_all.loc[mask, 'volume_new'] = (
                assign_all.loc[mask, 'volume_new'] + (sum_volume / 2) * (
                assign_all.loc[mask, 'volume_new'] / sum(assign_all.loc[mask, 'volume_new'])))
    return assign_all


def _recalc_speed(assign_all):
    """Recalculate BPR speed from volume/capacity ratio."""
    assign_all['vcratio'] = assign_all['volume_new'] / assign_all['capacity']
    assign_all['speed_bpr_new'] = assign_all.apply(
        lambda x: bpr_function(x.vcratio, x.ffs, x.alpha, x.beta), axis=1)
    return assign_all


def shift_peak(assign_all, r_rr):
    assign_all['volume_new'] = assign_all['volume_hourly']
    assign_all = _redistribute_peak(assign_all, [7, 8], 6, 9, r_rr)      # morning
    assign_all = _redistribute_peak(assign_all, [16, 17], 15, 18, r_rr)   # evening
    return _recalc_speed(assign_all)


def mode_shift(assign_all, r_rr):
    assign_all['volume_new'] = assign_all['volume_hourly'] * (1 - r_rr)
    return _recalc_speed(assign_all)


# ==================== Scenario dispatch ====================
SCENARIO_DISPATCH = {
    's_raw':    ('raw', 0),
    's_peak10': ('peak', 0.1), 's_peak20': ('peak', 0.2), 's_peak30': ('peak', 0.3),
    's_mode10': ('mode', 0.1), 's_mode20': ('mode', 0.2), 's_mode30': ('mode', 0.3),
}

MODE_SHIFT_RATIO = {
    's_mode10': 0.1, 's_mode20': 0.2, 's_mode30': 0.3,
}

# Camera vehicle column -> MOVES source type mapping
CAM_TO_SRC = {'c3_car_s': 21, 'c3_ptruck_s': 31, 'c3_htruck_s': 61,
              'c3_ltruck_s': 52, 'c3_sbus_s': 43, 'c3_bus_s': 42, 'c3_mcycle_s': 11}
ZERO_SRC_TYPES = ['32', '41', '51', '53', '54', '62']
P_C = list(CAM_TO_SRC.keys())

# ==================== Load data (once, outside loop) ====================
# Read signal network (template)
osm_mt_template = pd.read_pickle(r'D:\NY_Emission\Shp\osmdta_ritis_signal.pkl')
osm_mt_template['linkID'] = osm_mt_template['from_node_'].astype(str) + '_' + osm_mt_template['to_node_id'].astype(str)
osm_mt_template = osm_mt_template.drop_duplicates(subset=['linkID']).reset_index(drop=True)

# Read congestion pricing zone
geo_congest_price = gpd.read_file(r'D:\NY_Emission\Shp\congest_pricing.shp')

# Read speed volume change
mvolume = pd.read_pickle(r'D:\NY_Emission\CongestionPricing\volume_change.pkl')
mspeed = pd.read_pickle(r'D:\NY_Emission\CongestionPricing\speed_change.pkl')
mspeed['week'] = mspeed['week'] * 2 + 2

# Read vehicle registration (for age distribution)
veh_typet_raw = pd.read_csv(r'D:\NY_Emission\Volume_grth\Vehicle__Snowmobile__and_Boat_Registrations.csv')

# Read county
county_us = gpd.read_file(
    r'G:\Data\Dewey\SAFEGRAPH\Open Census Data\Census Website\2019\nhgis0011_shape\\US_county_2019.shp')

# Camera data (loaded once, avoids repeated API calls)
cams = pd.DataFrame(requests.get(r'https://webcams.nyctmc.org/api/cameras/').json())
cams_gpd_all = gpd.GeoDataFrame(cams, geometry=gpd.points_from_xy(x=cams.longitude, y=cams.latitude))
cams_gpd_all = cams_gpd_all.set_crs('EPSG:4326')

# Camera detection counts
count_df = pd.read_pickle(r'D:\NY_Emission\Video_Process\count_df_new_final.pkl')

# Weather (same for all scenarios since d_t is fixed)
weatherr = pd.read_excel(r'D:\NY_Emission\MOVES\Weather.xlsx')
weatherr['Date'] = pd.to_datetime(weatherr['Date'])
weatherr['Temperature'] = weatherr.Temperature.str.extract(r'(\d+)')
weatherr['Temperature'] = weatherr['Temperature'].astype(int).apply(lambda x: custom_round(x, base=5))
weatherr['Humidity'] = weatherr.Humidity.str.extract(r'(\d+)')
weatherr['Humidity'] = weatherr['Humidity'].astype(int).apply(lambda x: custom_round(x, base=5))

# Vehicle age distribution (same across scenarios)
veh_typet = veh_typet_raw[veh_typet_raw['Record Type'].isin(['VEH', 'TRL'])].reset_index(drop=True)
body_type_map = {
    21: ['4DSD', '2DSD', 'CONV', 'ATV', 'SEDN', 'TAXI'],
    31: ['SUBN', 'PICK', 'VAN'],
    42: ['BUS'],
    11: ['MCY'],
    54: ['H/WH', 'H/TR'],
    52: ['LTRL', 'UTIL', 'DELV'],
    61: ['SEMI', 'TRLR', 'DUMP', 'TRACTOR', 'FLAT', 'P/SH'],
}
veh_typet['sourceType'] = 21  # default
for src_type, body_types in body_type_map.items():
    veh_typet.loc[veh_typet['Body Type'].isin(body_types), 'sourceType'] = src_type
veh_typet['age'] = 2024 - veh_typet['Model Year']
veh_typet = veh_typet[(veh_typet['age'] <= 50) & (veh_typet['age'] >= 0)]
veh_age_dis = (veh_typet.groupby('sourceType')['age'].value_counts() /
               veh_typet.groupby('sourceType')['age'].count()).reset_index()
veh_age_dis.columns = ['sourceTypeID', 'ageID', 'ageFraction']
veh_age_dis['yearID'] = 2023

stl = [11, 21, 31, 32, 41, 42, 43, 51, 52, 53, 54, 61, 62]
veh_age_all = pd.DataFrame({'sourceTypeID': np.repeat(stl, 51), "ageID": list(range(0, 51)) * len(stl)})
veh_age_all = veh_age_all.merge(veh_age_dis, on=['sourceTypeID', "ageID"], how='left')
veh_age_all["ageFraction"] = veh_age_all.groupby(["ageID"])['ageFraction'].transform(
    lambda x: x.fillna(x.mean()))
veh_age_all['yearID'] = 2023

# Manhattan boundary (for Voronoi clipping)
county_mt = county_us[county_us['NAMELSAD'] == 'New York County']
county_mt = county_mt.to_crs('EPSG:32618')
county_mt['geometry'] = county_mt['geometry'].buffer(100)
county_mt = county_mt.to_crs('EPSG:4326')

# ==================== Scenario loop ====================
ess = ['s_mode10', 's_mode20', 's_mode30']
d_t = datetime.datetime(2023, 12, 5, 0, 0, 0)

for e_ess in tqdm(ess):
    print(e_ess)

    # --- Prepare network ---
    osm_mt = osm_mt_template.copy()
    osm_mt = osm_mt[['link_type_', 'linkID', 'tmc_code', 'Is_signal', 'geometry', 'Original', 'Cross',
                     'length', 'lanes', 'free_speed', 'capacity']]
    osm_mt['alpha'] = 0.15
    osm_mt['beta'] = 4
    osm_mt.loc[osm_mt['Original'] == 'motorway', 'ffs'] = 70
    osm_mt.loc[osm_mt['Original'] == 'primary', 'ffs'] = 60
    osm_mt.loc[osm_mt['Original'] == 'secondary', 'ffs'] = 40
    osm_mt.loc[osm_mt['Original'] == 'residential', 'ffs'] = 30
    osm_mt = osm_mt.to_crs(epsg=3857)
    osm_mt['linkLength'] = osm_mt.geometry.length  # meter

    geo_cp = geo_congest_price.to_crs(osm_mt.crs)
    sinct = gpd.sjoin(osm_mt, geo_cp, how="inner", predicate="intersects")
    osm_mt.loc[osm_mt['linkID'].isin(sinct['linkID']), 'is_cong'] = 1

    # --- Load assignment and compute BPR speed ---
    assign_all = pd.read_pickle(r'D:\NY_Emission\ODME_NY\Simulation_outcome\assign_all_.pkl')
    assign_all['linkID'] = assign_all['from_node_id'].astype(str) + '_' + assign_all['to_node_id'].astype(str)
    assign_all = assign_all.merge(osm_mt, on=['linkID'])
    assign_all['vcratio'] = assign_all['volume_hourly'] / assign_all['capacity']
    assign_all['speed_bpr'] = assign_all.apply(
        lambda x: bpr_function(x.vcratio, x.ffs, x.alpha, x.beta), axis=1)

    # --- Apply scenario ---
    scenario_type, ratio = SCENARIO_DISPATCH.get(e_ess, ('raw', 0))
    if scenario_type == 'raw':
        assign_all['speed_bpr_new'] = assign_all['speed_bpr']
        assign_all['volume_new'] = assign_all['volume_hourly']
    elif scenario_type == 'peak':
        assign_all = shift_peak(assign_all, ratio)
    elif scenario_type == 'mode':
        assign_all = mode_shift(assign_all, ratio)

    # --- Generate driving cycle (second-by-second speed) ---
    checkpoint_dir = Path(r'D:\NY_Emission\MOVES\input_%s' % e_ess)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for kk in tqdm(range(0, 24, 4)):
        speed1 = assign_all.loc[(assign_all['hour'] == kk),
            ['linkID', 'hour', 'speed_bpr_new', 'Is_signal', 'Original', 'Cross']].reset_index(drop=True)
        prev_length = len(speed1)
        speed1 = speed1.loc[speed1.index.repeat(3600)].reset_index(drop=True)
        speed1['measurement_tstamp'] = list(range(0, 3600)) * prev_length
        speed1['grade'] = 0
        speed1 = speed1.drop(['hour'], axis=1)
        speed1.columns = ['linkID', 'speed', 'Is_signal', 'Original', 'Cross', 'secondID', 'grade']
        print(f'{kk}: Length: {len(speed1)}')

        # Generate signal phase (Is_1): 1=green/moving, 0=red/stopped
        speed1 = generate_signal_phase(speed1)

        speed1['raw_speed'] = speed1['speed']
        speed1['speed'] = speed1['speed'] * speed1['Is_1']

        # Consider acc as 0.2g = 0.2*9.8*2.23694 mph/s = 4.385 mph/s
        speed1['cc_gp'] = (speed1['Is_1'] != speed1['Is_1'].shift()).cumsum()
        speed1['ccount'] = (speed1['speed'] / 4.385).astype(int)
        speed1['p_ctall'] = speed1.groupby(["cc_gp"])["linkID"].transform("count")
        speed1['p_ctallmin'] = ((speed1['p_ctall'] - 2) // 2).astype(int)
        speed1['ccount'] = speed1[['p_ctallmin', 'ccount']].min(axis=1)

        slice_per_group = pd.Series(speed1.ccount.values, index=speed1.cc_gp).to_dict()
        idx_head5 = speed1.groupby('cc_gp', group_keys=False).apply(
            lambda g: g.head(slice_per_group.get(g.name))).index
        idx_tail5 = speed1.groupby('cc_gp', group_keys=False).apply(
            lambda g: g.tail(slice_per_group.get(g.name))).index
        idx_head1 = speed1.groupby('cc_gp', group_keys=False).head(1).index
        idx_tail1 = speed1.groupby('cc_gp', group_keys=False).tail(1).index

        # Remove those without signal
        indxn = speed1[
            (~speed1['Is_signal']) | (speed1['Original'] == 'motorway') | (speed1['Cross'] == 'motorway')].index
        idx_head5 = idx_head5.difference(indxn)
        idx_tail5 = idx_tail5.difference(indxn)
        idx_head1 = idx_head1.difference(indxn)
        idx_tail1 = idx_tail1.difference(indxn)

        speed1['speed'] = speed1['speed'].ffill().bfill()
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

        print(f'{kk}: Length: {len(speed1)}')
        speed1[['linkID', 'secondID', 'speed', 'grade']].to_csv(
            r'D:\NY_Emission\MOVES\input_%s\drivingCycle_signal_%s.csv' % (e_ess, kk), index=False)

    # --- Generate link file (volume and speed) ---
    for kk in tqdm(range(0, 24, 4)):
        speedavg = assign_all[assign_all['hour'] == kk].groupby(['linkID'])['speed_bpr_new'].mean().reset_index()
        volumeavg = assign_all[assign_all['hour'] == kk].groupby(['linkID'])['volume_new'].mean().reset_index()
        speed2 = speedavg.merge(volumeavg, on=['linkID'], how='left')
        speed2 = speed2.merge(osm_mt[['linkID', 'linkLength']], on=['linkID'], how='left')
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

    # --- Source type distribution ---
    ct_weight = count_df.groupby(['file', 'Hour'])['frame'].count() / 3600
    ct_each = count_df.groupby(['file', 'Hour'])[P_C].sum().div(ct_weight, axis=0).reset_index()
    ct_each['id'] = ct_each['file'].str.replace('.mp4', '')
    ct_each['sum'] = ct_each[P_C].sum(axis=1)

    # Mode shift: reduce cars, increase buses
    if e_ess in MODE_SHIFT_RATIO:
        r = MODE_SHIFT_RATIO[e_ess]
        ct_each['c3_car_s'] = ct_each['c3_car_s'] * (1 - r)
        ct_each['c3_bus_s'] = ct_each['c3_bus_s'] * (1 + r / 30)

    avg_ratio = pd.DataFrame(ct_each[P_C].sum() / ct_each['sum'].sum()).reset_index()
    avg_ratio.columns = ['sourceTypeID', 'ratio']
    avg_ratio = avg_ratio.replace({'sourceTypeID': CAM_TO_SRC})
    avg_ratio = pd.concat([avg_ratio, pd.DataFrame({'sourceTypeID': ZERO_SRC_TYPES,
                                                    'ratio': [0] * len(ZERO_SRC_TYPES)})], axis=0)
    avg_ratio['sourceTypeID'] = avg_ratio['sourceTypeID'].astype(int)

    for kk in P_C:
        ct_each[kk] = ct_each[kk] / ct_each['sum']
    for kk in ZERO_SRC_TYPES:
        ct_each[kk] = 0
    ct_each = pd.melt(ct_each, id_vars=['id', 'Hour'], value_vars=P_C + ZERO_SRC_TYPES)
    ct_each = ct_each.replace({'variable': CAM_TO_SRC})
    ct_each['variable'] = ct_each['variable'].astype(int)
    ct_each = ct_each.sort_values(by=['id', 'Hour', 'variable'], ascending=True).reset_index(drop=True)

    # Voronoi polygon for camera-link matching
    cams_gpd = cams_gpd_all[cams_gpd_all['id'].isin(ct_each['id'])].reset_index(drop=True)
    coords = np.vstack((cams_gpd.geometry.x.values, cams_gpd.geometry.y.values)).T
    vor = Voronoi(coords)
    lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
    polys = shapely.ops.polygonize(lines)
    voronois = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys)).set_crs('EPSG:4326')
    voronois = gpd.overlay(voronois, county_mt).reset_index()

    # Sjoin camera -> Voronoi -> road link
    SInBG = gpd.sjoin(cams_gpd, voronois, how='inner', predicate='within').reset_index(drop=True)
    SInBG_index = SInBG[['id', 'index']].drop_duplicates(subset='index', keep='first')
    voronois = voronois.merge(SInBG_index, on='index', how='left')

    voronois = voronois.to_crs('EPSG:3857')
    SInBG = gpd.sjoin(osm_mt, voronois, how='left', predicate='intersects').reset_index(drop=True)
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
        ct_each1.to_csv(r'D:\NY_Emission\MOVES\input_%s\sourceTypeDistribution_NYC_%s.csv' % (e_ess, kk), index=False)

    # --- Vehicle age distribution (same for all scenarios) ---
    veh_age_all.to_csv(r'D:\NY_Emission\MOVES\input_%s\ageDistribution_2023.csv' % e_ess, index=False)

    # --- Meteorology ---
    weather = pd.DataFrame(
        {'monthID': [d_t.month] * 24, 'zoneID': [36061] * 24, "hourID": range(0, 24),
         "temperature": weatherr.loc[weatherr['Date'] == d_t, 'Temperature'].to_list(),
         "realHumidity": weatherr.loc[weatherr['Date'] == d_t, 'Humidity'].to_list()})
    weather.to_csv(r'D:\NY_Emission\MOVES\input_%s\meteorology_NYC_36061_01.csv' % e_ess, index=False)

    # --- Generate batchmode ---
    batchmode = pd.DataFrame({
        'taskID': range(0, 24, 4),
        'region': 'newyork',
        'calendarYear': 2023,
        'meteorologyFileName': 'meteorology_NYC_36061_01.csv',
        'sourceTypeDistributionFileName': ['sourceTypeDistribution_NYC_%s.csv' % ii for ii in range(0, 24, 4)],
        'ageDistributionFileName': 'ageDistribution_2023.csv',
        'linkFileName': ['link_NYC_36061_%s.csv' % ii for ii in range(0, 24, 4)],
        'driveSchedule/OpModeDistributionFileName': ['drivingCycle_signal_%s.csv' % ii for ii in range(0, 24, 4)],
        'opmode(o)/cycle(d)/speed(v)': 'd',
    })
    batchmode.to_csv(r'D:\NY_Emission\MOVES\input_%s\batchmode.csv' % e_ess, index=False)