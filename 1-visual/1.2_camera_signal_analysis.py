import datetime
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import contextily as ctx
import seaborn as sns
import matplotlib as mpl
import requests
import json
from scipy.spatial import Voronoi
import shapely
from shapely import geometry, ops

sns.set_palette("coolwarm")

pd.options.mode.chained_assignment = None


def split_data(count_df, v_n):
    count_df[[v_n + '_car', v_n + '_truck', v_n + '_bus', v_n + '_cycle']] = pd.DataFrame(
        count_df[v_n].str.strip('[]').str.split(',').to_list()).astype(int)
    count_df[v_n + '_total'] = count_df[[v_n + '_car', v_n + '_truck', v_n + '_bus', v_n + '_cycle']].sum(axis=1)
    count_df[v_n + '_total_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_total'].diff(-1) * (-1)
    count_df[v_n + '_car_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_car'].diff(-1) * (-1)
    count_df[v_n + '_truck_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_truck'].diff(-1) * (-1)
    count_df[v_n + '_bus_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_bus'].diff(-1) * (-1)
    count_df[v_n + '_cycle_s'] = count_df.groupby(['file', 'Date_hour'])[v_n + '_cycle'].diff(-1) * (-1)
    return count_df


# Match signal info with road network
# Read network data
osm = gpd.read_file(r'D:\\NY_Emission\Shp\osmdta_ritis.shp')
osm = osm.to_crs('EPSG:32618')

# Cut by Manhattan
county_us = gpd.read_file(r'G:\Data\SafeGraph\Open Census Data\Census Website\2019\nhgis0011_shape\\US_county_2019.shp')
county_mt = county_us[county_us['NAMELSAD'] == 'New York County']
county_mt = county_mt.to_crs('EPSG:32618')
county_mt['geometry'] = county_mt['geometry'].buffer(100)
osm_mt = gpd.sjoin(osm, county_mt, how='inner', op='intersects')
osm_mt = osm_mt[~osm_mt['STATEFP'].isnull()].reset_index(drop=True)
# osm_mt = gpd.overlay(osm, county_us, how='union').explode().reset_index(drop=True)

# Read intersection info
intsec = gpd.read_file(r'D:\\NY_Emission\Shp\OSM\new-york-latest-free.shp\gis_osm_traffic_free_1.shp')
intsec = intsec.to_crs('EPSG:32618')
intsec_mta = gpd.sjoin(intsec, county_mt, how='inner')
intsec_mt = intsec_mta[intsec_mta['fclass'] == 'traffic_signals'].reset_index(drop=True)
intsec_mta['fclass'].value_counts()
# Attach the intersection to road
osm_mt['Is_signal'] = False
osm_mt['Cross_fclass'] = np.nan
intsec_mt['Cross_fclass'] = np.nan
osm_mt['Cross'] = np.nan
osm_mt['Original'] = osm_mt['link_type_'].replace(
    {"unclassified": "residential", "tertiary": "residential", "trunk": "motorway"})
# osm_mt['Cross_fclass2'] = np.nan
# intsec_mt[intsec_mt['osm_id'] == '4016646206']
for jj in tqdm(range(0, len(intsec_mt))):
    tem_intsec = intsec_mt.loc[jj,]
    # Find the smallest distance
    osm_mt.loc[:, 'Distance'] = [tem_intsec.geometry.distance(var) for var in osm_mt.geometry]  # m
    join_link = osm_mt[osm_mt['Distance'] < 0.1]
    if len(join_link) > 0:
        intsec_mt.loc[jj, ['Cross_fclass']] = [set(join_link['Original'])]

        osm_mt.loc[join_link.index, 'Cross_fclass'] = [set(join_link['Original'])] * len(join_link)
        osm_mt.loc[join_link.index, 'Is_signal'] = True
        # break
        for rr in join_link.index:
            other_link = join_link.drop(rr, axis=0)
            other_link['D_Direction'] = abs(other_link['Direction'] - join_link.loc[rr, 'Direction'])
            other_link = other_link[((other_link['D_Direction'] > 20) & (other_link['D_Direction'] < 160)) | (
                    (other_link['D_Direction'] > 200) & (other_link['D_Direction'] < 340))]
            if len(other_link) > 0:
                osm_mt.loc[rr, 'Cross'] = list(other_link['Original'])[0]

osm_mt.to_pickle(r'D:\\NY_Emission\Shp\osmdta_ritis_signal.pkl')
intsec_mt.to_pickle(r'D:\\NY_Emission\Shp\intersect_signal_timing.pkl')

# Plot signal spatial distribution
osm_mt = pd.read_pickle(r'D:\NY_Emission\Shp\osmdta_ritis_signal.pkl')
intsec_mt = pd.read_pickle(r'D:\NY_Emission\Shp\intersect_signal_timing.pkl')

# Calculate the intersection between each pair of geometries
intersection_points = osm_mt.geometry.intersection(osm_mt.geometry)
intersection_points.plot()

# Plot intersection time
intsec_mt['Cross_fclass_s'] = intsec_mt['Cross_fclass'].astype(str)
intsec_mt['Cross_fclass_s'].value_counts()
intsec_mt['Cross_fclass_s'] = intsec_mt['Cross_fclass_s'].replace(
    {"{'motorway', 'primary'}": 'primary-motorway', "{'primary', 'motorway'}": 'primary-motorway',
     "{'primary', 'residential'}": 'primary-residential', "{'primary', 'secondary'}": 'primary-secondary',
     "{'secondary', 'primary'}": 'primary-secondary', "{'primary'}": 'primary-primary',
     "{'residential', 'secondary'}": 'residential-secondary', "{'secondary', 'residential'}": 'residential-secondary',
     "{'residential'}": 'residential-residential', "{'secondary'}": 'secondary-secondary'})
intsec_mt = intsec_mt.dropna(subset='CTime')
intsec_mt = intsec_mt[intsec_mt['Cross_fclass_s'] != 'primary-motorway']
intsec_mt.groupby('Cross_fclass_s')['CTime'].mean()
intsec_mt.groupby('Cross_fclass_s')['CTime'].std()
sns.set_palette(sns.color_palette('coolwarm', 6))
fig, ax = plt.subplots(figsize=(4.5, 5.5))
sns.barplot(intsec_mt[['Cross_fclass_s', 'CTime']], y='Cross_fclass_s', x='CTime',
            palette=sns.color_palette('coolwarm', 6))
plt.ylabel('')
plt.xlabel('Cycle length (s)')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\cycle_length.pdf')

# Plot spatial map
fig, ax = plt.subplots(figsize=(4, 8))
osm_mt.plot(ax=ax, alpha=0.3, lw=1, color='white')
mpl.rcParams['text.color'] = 'white'
intsec_mt.plot(column="CTime", cmap='coolwarm', scheme="natural_breaks", k=4, markersize=6, ax=ax, alpha=0.8,
               legend=True, legend_kwds={'labelcolor': 'w', "fmt": "{:.0f}", 'title': 'Cycle length (s)'})
plt.xlim(intsec_mt.bounds['minx'].min(), intsec_mt.bounds['maxx'].max())
plt.ylim(intsec_mt.bounds['miny'].min(), intsec_mt.bounds['maxy'].max())
ctx.add_basemap(ax, crs=osm_mt.crs, source=ctx.providers.CartoDB.DarkMatter, alpha=0.9)
plt.subplots_adjust(top=0.995, bottom=0.005, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
plt.axis('off')
mpl.rcParams['text.color'] = 'k'
# plt.show()
plt.savefig(r'D:\NY_Emission\Figure\Signal_Timing.pdf')

# Plot TS polyline
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
cams_gpd = cams_gpd[cams_gpd['area'] == 'Manhattan']

# Generate a Thiessen Polygon
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

fig, ax = plt.subplots(figsize=(3.5, 7))
mpl.rcParams['text.color'] = 'white'
cams_gpd.plot(ax=ax, alpha=0.9,markersize=6)
voronois.boundary.plot(ax=ax, color='r', lw=1, alpha=1)
mpl.rcParams['text.color'] = 'k'
# voronois.boundary.plot(ax=ax, color='gray', lw=0.05, alpha=0.5)
ctx.add_basemap(ax, crs=county_mt.crs, source=ctx.providers.CartoDB.DarkMatter, alpha=0.9)
plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
# plt.tight_layout()
plt.axis('off')
mpl.rcParams['text.color'] = 'k'
# plt.show()
plt.savefig(r'D:\NY_Emission\Figure\TS_polyline.pdf')

# Plot signal control demo
allcount = pd.read_csv(
    r'D:\NY_Emission\Video_Process\data_models\green_light_count_1572a83a-0a4f-4a7b-84a0-fec0890a2de3.mp4_2023_12_05_11.csv')
for kk in ['c1', 'c2', 'c3', 'c4']: allcount = split_data(allcount, kk)
allcount['green_1'] = 0
allcount.loc[allcount['l_color'] == 'green', 'green_1'] = 1
allcount.loc[allcount['l_color'].isnull(), 'green_1'] = np.nan
allcount['green_1'] = allcount['green_1'].fillna(method='ffill')

g = allcount['green_1'].ne(allcount['green_1'].shift()).cumsum()
allcount['Count'] = allcount.groupby(g)['green_1'].transform('size') * np.where(allcount['green_1'], 1, -1)
allcount.loc[allcount['Count'].abs() < 3, 'green_1'] = np.nan
allcount['green_1'] = allcount['green_1'].fillna(method='ffill')
allcount.loc[allcount['green_1'] == 0, 'c3_total_s'] = 0

allcount = allcount[(allcount['Unnamed: 0'] > 480) & (allcount['Unnamed: 0'] < 1100 - 200)].reset_index(drop=True)
allcount.index = allcount.index * 2

fig, axs = plt.subplots(figsize=(5, 5), nrows=2, ncols=1, sharex=True)
# allcount['num_repeat'].diff().plot(ax=axs[0])
allcount['c3_total_s'].plot(ax=axs[0])
axt = axs[0].twinx()
allcount['green_1'].plot(ax=axt, color='green')
axs[0].set_ylabel('Arrival')

allcount['num_repeat'].diff().plot(ax=axs[1])
axt = axs[1].twinx()
(allcount['green_1'] == 0).astype(int).plot(ax=axt, color='red')
axs[1].set_ylabel('Queuing')
axs[1].set_xlabel('Second')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\green_light.pdf')


# Demo: Transfer signal into driving cycle
raw_sd = pd.read_csv(r'D:\NY_Emission\Speed\NY_TT1\NY_TT1.csv')
raw_sd['measurement_tstamp'] = pd.to_datetime(raw_sd['measurement_tstamp'])
raw_sd1 = raw_sd[(raw_sd['tmc_code'] == '120+05434') & (raw_sd['measurement_tstamp'].dt.hour == 18) & (
        raw_sd['measurement_tstamp'].dt.date == datetime.date(2023, 12, 4))].reset_index(drop=True)
raw_sd1 = raw_sd1.set_index('measurement_tstamp').asfreq('s', method='ffill').reset_index()
raw_sd1['seconds'] = raw_sd1['measurement_tstamp'].dt.hour * 3600 + raw_sd1['measurement_tstamp'].dt.minute * 60 + \
                     raw_sd1['measurement_tstamp'].dt.second
# 60s+30s
raw_sd1['30s'] = raw_sd1['seconds'] // 30 - min(raw_sd1['seconds'] // 30)
raw_sd1['30s'] = raw_sd1['30s'] % 3
raw_sd1['90s'] = raw_sd1['seconds'] // 90
raw_sd1['new_speed'] = np.nan
raw_sd1.loc[raw_sd1['30s'] == 2, 'new_speed'] = raw_sd1.loc[raw_sd1['30s'] == 2, 'speed']
raw_sd1['loss'] = raw_sd1.groupby(['90s'])['new_speed'].transform('sum') / 60
raw_sd1.loc[raw_sd1['30s'] == 2, 'new_speed'] = 0
raw_sd1.loc[raw_sd1['30s'] != 2, 'new_speed'] = raw_sd1.loc[raw_sd1['30s'] != 2, 'speed'] + raw_sd1.loc[
    raw_sd1['30s'] != 2, 'loss']
all_list = []
for kk in [0, 1, 2, 3, 4, 25, 26, 27, 28, 29]: all_list.extend(raw_sd1.loc[raw_sd1['30s'] == 2].index[kk::30])
raw_sd1.loc[all_list, 'new_speed'] = np.nan
raw_sd1['new_speed'] = raw_sd1['new_speed'].interpolate(method='polynomial', order=2)
raw_sd1['new_second'] = (raw_sd1['measurement_tstamp'] - raw_sd1['measurement_tstamp'].min()).dt.total_seconds()
raw_sd1 = raw_sd1[raw_sd1['new_second'] <= 1800]
# Plot
fig, ax = plt.subplots(figsize=(4, 4), nrows=2, ncols=1, sharex=True)
ax[0].plot(raw_sd1['new_second'], raw_sd1['speed'])
ax[0].set_ylabel('Speed (raw)')
ax[0].set_xlabel('Second')
ax[1].plot(raw_sd1['new_second'], raw_sd1['new_speed'])
ax[1].set_ylabel('Speed (new)')
ax[1].set_xlabel('Second')
ax[0].axvline(1240, color='r', linestyle='--')
ax[0].axvline(1330, color='r', linestyle='--')
ax[1].axvline(1240, color='r', linestyle='--')
ax[1].axvline(1330, color='r', linestyle='--')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\Speed_convert.pdf')

fig, ax = plt.subplots(figsize=(4, 4), nrows=2, ncols=1, sharex=True)
ax[0].plot(raw_sd1['new_second'], raw_sd1['speed'])
ax[0].set_ylabel('Speed (raw)')
ax[0].set_xlabel('Second')
ax[1].plot(raw_sd1['new_second'], raw_sd1['new_speed'])
ax[1].set_ylabel('Speed (new)')
ax[1].set_xlabel('Second')
plt.xlim([1240, 1330])
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\Speed_convert_small.pdf')
