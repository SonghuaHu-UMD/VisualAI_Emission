"""
Shared utility functions for the VisualAI Emission project.
Eliminates code duplication across pipeline scripts.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from scipy.spatial import Voronoi
import shapely
from shapely import geometry, ops

from config import (OSM_SIGNAL_PKL, CAMERA_API_URL, COUNTY_SHP,
                    EPSG_WGS84, BPR_ALPHA_DEFAULT, BPR_BETA_DEFAULT,
                    FFS_BY_ROAD_TYPE, MPH_TO_MPS)


# ==================== Road Network ====================

def load_road_network(pkl_path=OSM_SIGNAL_PKL, columns=None):
    """Load the OSM road network with signal data, create linkID, and deduplicate."""
    osm_mt = pd.read_pickle(pkl_path)
    osm_mt['linkID'] = osm_mt['from_node_'].astype(str) + '_' + osm_mt['to_node_id'].astype(str)
    osm_mt = osm_mt.drop_duplicates(subset=['linkID']).reset_index(drop=True)
    if columns is not None:
        osm_mt = osm_mt[columns]
    return osm_mt


def load_county_boundary(county_name='New York County', buffer_m=100, target_crs=None):
    """Load county boundary, optionally buffer and reproject."""
    county_us = gpd.read_file(COUNTY_SHP)
    county_mt = county_us[county_us['NAMELSAD'] == county_name]
    county_mt = county_mt.to_crs('EPSG:32618')
    if buffer_m > 0:
        county_mt = county_mt.copy()
        county_mt['geometry'] = county_mt['geometry'].buffer(buffer_m)
    if target_crs:
        county_mt = county_mt.to_crs(target_crs)
    return county_mt, county_us


# ==================== Camera Data ====================

def load_camera_data(filter_ids=None):
    """Load camera locations from NYC TMC API."""
    cams = pd.DataFrame(requests.get(CAMERA_API_URL).json())
    cams_gpd = gpd.GeoDataFrame(cams, geometry=gpd.points_from_xy(x=cams.longitude, y=cams.latitude))
    cams_gpd = cams_gpd.set_crs(EPSG_WGS84)
    if filter_ids is not None:
        cams_gpd = cams_gpd[cams_gpd['id'].isin(filter_ids)].reset_index(drop=True)
    return cams_gpd


def build_voronoi_polygons(points_gpd, clip_boundary=None):
    """Generate Thiessen (Voronoi) polygons from point GeoDataFrame."""
    x = points_gpd.geometry.x.values
    y = points_gpd.geometry.y.values
    coords = np.vstack((x, y)).T
    vor = Voronoi(coords)
    lines = [shapely.geometry.LineString(vor.vertices[line])
             for line in vor.ridge_vertices if -1 not in line]
    polys = shapely.ops.polygonize(lines)
    voronois = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys))
    voronois = voronois.set_crs(points_gpd.crs)
    if clip_boundary is not None:
        clip_boundary = clip_boundary.to_crs(voronois.crs)
        voronois = gpd.overlay(voronois, clip_boundary)
    voronois = voronois.reset_index()
    return voronois


# ==================== Traffic Functions ====================

def bpr_function(vc_ratio, ffs, alpha=BPR_ALPHA_DEFAULT, beta=BPR_BETA_DEFAULT):
    """Bureau of Public Roads (BPR) speed-flow function.

    Args:
        vc_ratio: Volume-to-capacity ratio
        ffs: Free-flow speed
        alpha: BPR alpha parameter (default: 0.15 per HCM)
        beta: BPR beta parameter (default: 4 per HCM)

    Returns:
        Estimated speed under given V/C ratio
    """
    return ffs / (1 + alpha * np.power(vc_ratio, beta))


def density_speed_function(density, ffs, k_critical, mm):
    """Fundamental diagram model: density-speed function."""
    x_over_k = density / k_critical
    dominator = 1 + np.power(x_over_k, mm)
    order = 2 / mm
    return ffs / np.power(dominator, order)


def assign_ffs_by_road_type(df, road_type_col='Original', ffs_col='ffs'):
    """Assign free-flow speed based on road type."""
    for road_type, ffs in FFS_BY_ROAD_TYPE.items():
        df.loc[df[road_type_col] == road_type, ffs_col] = ffs
    return df


# ==================== Data Parsing ====================

def split_data_yolo(count_df, v_n):
    """Parse YOLO detection counts (4 types: car, truck, bus, cycle)."""
    count_df[[v_n + '_car', v_n + '_truck', v_n + '_bus', v_n + '_cycle']] = pd.DataFrame(
        count_df[v_n].str.strip('[]').str.split(',').to_list()).astype(int)
    count_df[v_n + '_total'] = count_df[[v_n + '_car', v_n + '_truck', v_n + '_bus', v_n + '_cycle']].sum(axis=1)
    for suffix in ['total', 'car', 'truck', 'bus', 'cycle']:
        col = f'{v_n}_{suffix}'
        count_df[f'{col}_s'] = count_df.groupby(['file', 'Date_hour'])[col].diff(-1) * (-1)
    return count_df


def split_data_own(count_df, v_n):
    """Parse custom model detection counts (8 types)."""
    type_suffixes = ['car', 'ptruck', 'ltruck', 'htruck', 'sbus', 'bus', 'mcycle', 'truck']
    col_names = [f'{v_n}_{s}' for s in type_suffixes]
    count_df[col_names] = pd.DataFrame(
        count_df[v_n].str.strip('[]').str.split(',').to_list()).astype(int)
    count_df[f'{v_n}_total'] = count_df[col_names].sum(axis=1)
    for col in [f'{v_n}_total'] + col_names:
        count_df[f'{col}_s'] = count_df.groupby(['file', 'Date_hour'])[col].diff(-1) * (-1)
    return count_df


def split_data_by_type(count_df, y_var='peds', f_name='car_st', v_name='person'):
    """Extract count of a specific object type from detection results."""
    count_df[y_var] = count_df[y_var].astype(str)
    count_df[f_name + '_i'] = count_df[y_var].str.find(v_name)
    count_df[f_name] = count_df[y_var].apply(
        lambda st: st[(st.find(v_name) + len(v_name) + 3):(st.find(v_name) + len(v_name) + 6)])
    count_df[f_name] = count_df[f_name].str.extract(r'(\d+)', expand=False).astype(float)
    count_df.loc[count_df[f_name + '_i'] == -1, f_name] = 0
    count_df[f_name] = count_df[f_name].fillna(0)
    count_df = count_df.drop(f_name + '_i', axis=1)
    return count_df


# ==================== Geometry ====================

def line_direction(link, from_north=False):
    """Calculate direction angle of line geometries."""
    link['Start_Lon_utm'] = link['geometry'].apply(lambda g: g.coords[0][0])
    link['Start_Lat_utm'] = link['geometry'].apply(lambda g: g.coords[0][1])
    link['End_Lon_utm'] = link['geometry'].apply(lambda g: g.coords[-1][0])
    link['End_Lat_utm'] = link['geometry'].apply(lambda g: g.coords[-1][1])
    orig = np.asarray([link['Start_Lon_utm'], link['Start_Lat_utm']])
    dest = np.asarray([link['End_Lon_utm'], link['End_Lat_utm']])
    dx, dy = dest - orig
    ang = np.degrees(np.arctan2(dy, dx))
    if from_north:
        ang = np.mod((450.0 - ang), 360.)
    return ang


# ==================== Miscellaneous ====================

def generate_signal_phase(speed_df, signal_pkl_path=None):

    if signal_pkl_path is None:
        signal_pkl_path = r'D:\NY_Emission\Shp\intersect_signal_timing.pkl'

    # Load signal timing data
    try:
        intsec_mt = pd.read_pickle(signal_pkl_path)
        # Get average cycle time by cross road type
        intsec_mt['Cross_fclass_s'] = intsec_mt['Cross_fclass'].astype(str)
        cycle_times = intsec_mt.dropna(subset=['CTime']).groupby('Cross_fclass_s')['CTime'].mean()
        avg_cycle_time = intsec_mt['CTime'].dropna().mean()
    except (FileNotFoundError, KeyError):
        avg_cycle_time = 90  # default cycle length in seconds
        cycle_times = pd.Series()

    # Default: all green (Is_1 = 1)
    speed_df['Is_1'] = 1

    # For signalized links: create red/green pattern
    signal_mask = speed_df['Is_signal'].astype(bool)
    if signal_mask.any():
        # Assume green ratio ~0.5 (symmetric split), use cycle time to create pattern
        # Green phase = first half of cycle, Red phase = second half
        cycle_length = int(avg_cycle_time) if avg_cycle_time > 0 else 90
        green_time = cycle_length // 2

        # For signalized links, assign phase based on secondID within cycle
        second_in_cycle = speed_df.loc[signal_mask, 'secondID'].astype(int) % cycle_length
        speed_df.loc[signal_mask, 'Is_1'] = (second_in_cycle < green_time).astype(int).values

    return speed_df


def custom_round(x, base=5):
    """Round to nearest multiple of base."""
    return int(base * round(float(x) / base))


def read_emission_files(pattern, pollutant_map, unit_factor=0.000621371):
    """Read and concatenate emission output files.

    Args:
        pattern: Glob pattern for emission CSV files
        pollutant_map: Dict mapping pollutant IDs to names
        unit_factor: Conversion factor (default: meter to mile)
    """
    import glob
    import re
    all_files = glob.glob(pattern)
    emission_df = pd.DataFrame()
    for kk in all_files:
        df = pd.read_csv(kk)
        t_hour = re.findall(r'\d+', kk)
        df['Hour'] = t_hour[1] if len(t_hour) > 1 else t_hour[0]
        df['emquant'] = df['emquant'] * unit_factor
        emission_df = pd.concat([emission_df, df])
    if len(emission_df) > 0:
        emission_df['Hour'] = emission_df['Hour'].astype(int)
        emission_df = emission_df.sort_values(by=['linkID', 'pollutantID', 'Hour'])
        emission_df = emission_df.replace({'pollutantID': pollutant_map})
    return emission_df