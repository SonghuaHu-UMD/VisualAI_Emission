"""
Centralized configuration for the VisualAI Emission project.
Contains shared paths, constants, mappings, and plot styles.
"""

# ==================== Data Paths ====================
# Adjust these paths based on your environment (D: or E: drive)
DATA_ROOT = r'D:\NY_Emission'
DATA_ROOT_E = r'E:\NY_Emission'

# Sub-paths
SHP_DIR = rf'{DATA_ROOT}\Shp'
MOVES_DIR = rf'{DATA_ROOT}\MOVES'
VIDEO_DIR = rf'{DATA_ROOT}\Video_Process'
ODME_DIR = rf'{DATA_ROOT}\ODME_NY'
FIGURE_DIR = rf'{DATA_ROOT}\Figure'
SPEED_DIR = rf'{DATA_ROOT}\Speed'
VOLUME_DIR = rf'{DATA_ROOT}\Volume_grth'

# Key files
OSM_SIGNAL_PKL = rf'{SHP_DIR}\osmdta_ritis_signal.pkl'
COUNTY_SHP = r'G:\Data\Dewey\SAFEGRAPH\Open Census Data\Census Website\2019\nhgis0011_shape\US_county_2019.shp'
CAMERA_API_URL = r'https://webcams.nyctmc.org/api/cameras/'

# ==================== Constants ====================
COUNTY_ID = 36061
ZONE_ID = 360610
EPSG_WGS84 = 'EPSG:4326'
EPSG_UTM18N = 'EPSG:32618'
EPSG_WEB_MERCATOR = 'EPSG:3857'

# Study area counties
CT_L = ['36061', '36005', '34003', '34017', '36081', '35047', '36047', '36119', '36087']
CT_NY = ['36047', '36061', '36081', '36085', '36005']

# ==================== MOVES Source Types ====================
# Motorcycles: 11; Passenger Cars: 21; Passenger Trucks: 31; Light Commercial Trucks: 32
# Other Buses: 41; Transit Buses: 42; School Buses: 43; Refuse Trucks: 51
# Short-Haul Single Unit Trucks: 52; Long-Haul Single Unit Trucks: 53; Motor Homes: 54
# Short-Haul Combination Trucks: 61; Long-Haul Combination Trucks: 62
SOURCE_TYPES = [11, 21, 31, 32, 41, 42, 43, 51, 52, 53, 54, 61, 62]

SOURCE_TYPE_NAMES = {
    '11': 'Motorcycle', '21': 'Passenger Car', '31': 'Passenger Truck',
    '42': 'Transit Bus', '43': 'School Bus', '52': 'Single-Unit Truck',
    '61': 'Combination Truck'
}

# Camera detection -> MOVES source type mapping
CAMERA_TO_SOURCE_TYPE = {
    'c3_car_s': 21, 'c3_ptruck_s': 31, 'c3_htruck_s': 61,
    'c3_ltruck_s': 52, 'c3_sbus_s': 43, 'c3_bus_s': 42, 'c3_mcycle_s': 11
}

# Source types with zero ratio (not detected by camera)
ZERO_SOURCE_TYPES = ['32', '41', '51', '53', '54', '62']

# Camera vehicle count columns
CAMERA_VEHICLE_COLS = ['c3_car_s', 'c3_ptruck_s', 'c3_htruck_s', 'c3_ltruck_s',
                       'c3_sbus_s', 'c3_bus_s', 'c3_mcycle_s']

# ==================== Pollutant Mapping ====================
POLLUTANT_MAP = {
    1: 'TGH', 2: 'CO', 3: 'NOx', 87: 'VOC', 90: 'CO2',
    91: 'TEC', 98: 'CO2E', 100: 'PM10', 110: 'PM2.5'
}

TARGET_POLLUTANTS = ['CO', 'NOx', 'CO2', 'PM2.5']

# ==================== BPR Function Parameters ====================
# Standard BPR parameters (HCM reference)
BPR_ALPHA_DEFAULT = 0.15
BPR_BETA_DEFAULT = 4

# Free-flow speed by road type (mph)
FFS_BY_ROAD_TYPE = {
    'motorway': 70,
    'primary': 60,
    'secondary': 40,
    'residential': 30
}

# ==================== Unit Conversion ====================
METER_TO_MILE = 0.000621371
MPH_TO_MPS = 0.44704  # miles per hour to meters per second

# ==================== Plot Style ====================
PLOT_STYLE = {
    'font.size': 15, 'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
    'xtick.direction': 'in', 'xtick.major.size': 0.5,
    'grid.linestyle': '--', 'axes.grid': True, 'grid.alpha': 1, 'grid.color': '#cccccc',
    'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
    'ytick.direction': 'in', 'ytick.major.size': 0.5,
    'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5, 'ytick.minor.visible': True, 'ytick.right': True,
    'axes.linewidth': 0.5, 'grid.linewidth': 0.5, 'lines.linewidth': 1.5,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05
}