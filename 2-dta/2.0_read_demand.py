import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import seaborn as sns
from tqdm import tqdm
import os
import datetime
from functools import reduce

pd.options.mode.chained_assignment = None
CT_L = ['36061', '36005', '34003', '34017', '36081', '35047', '36047', '36119', '36087']
CT_NY = ['36047', '36061', '36081', '36085', '36005']
temp_n = ['325120', '447190', '454310', '541614', '336320', '424720', '221111', '221112', '221113', '221118']

## 1. Read event OD From MDLD ##
f_url = r'D:\NY_Emission\ODME_NY\OD_File\MDLD'
full_od = pd.concat([pd.read_csv(r"%s\full_od_hour_ny_1.csv" % f_url),
                     pd.read_csv(r"%s\full_od_hour_ny_2.csv" % f_url)], axis=0)
full_od['timestamp'] = pd.to_datetime(full_od['hour'], errors='coerce', utc=True)
full_od['timestamp'] = full_od['timestamp'].dt.tz_convert('US/Eastern')
full_od = full_od[~full_od['timestamp'].isnull()]
full_od['hour'] = full_od['timestamp'].dt.hour
full_od['dayofweek'] = full_od['timestamp'].dt.dayofweek
full_od['date'] = full_od['timestamp'].dt.date

# Change to CTFIPS
ct_rp = {'CT.': '09', 'MA.': '25', 'NJ.': '34', 'NY.': '36', 'PA.': '42', 'US.': ''}
full_od['start_admin2'] = full_od['start_admin2'].replace(ct_rp, regex=True)
full_od['end_admin2'] = full_od['end_admin2'].replace(ct_rp, regex=True)

# Four events: snow storm (2022/1/29); covid (2020/3/21); Thanksgiving Eve (2021/11/24); Henri flooding (2021/8/22);
# average OD FLOW in hour in county
full_od_n = full_od[(full_od['start_admin2'].isin(CT_L)) & (full_od['end_admin2'].isin(CT_L))].reset_index(drop=True)
full_od_n.to_pickle(r'D:\\NY_Emission\ODME_NY\OD_File\MDLD\od_daily_ratio.pkl')
full_od_avg = full_od_n[(full_od_n['timestamp'].dt.date <= datetime.date(2021, 8, 14)) & (
        full_od_n['timestamp'].dt.date >= datetime.date(2021, 8, 1))]
full_od_avg = full_od_avg.groupby(['start_admin2', 'end_admin2', 'hour'])['total_trips'].mean().reset_index()
full_od_avg.columns = ['start_admin2', 'end_admin2', 'hour', 'total_trips_avg']
print(full_od_avg['total_trips'].sum())

full_od_ss = full_od_n[full_od_n['timestamp'].dt.date == datetime.date(2022, 1, 29)]
full_od_ss = full_od_ss.groupby(['start_admin2', 'end_admin2', 'hour'])['total_trips'].mean().reset_index()
full_od_ss.columns = ['start_admin2', 'end_admin2', 'hour', 'total_trips_ss']
print(full_od_ss['total_trips'].sum())

full_od_hf = full_od_n[full_od_n['timestamp'].dt.date == datetime.date(2021, 8, 22)]
full_od_hf = full_od_hf.groupby(['start_admin2', 'end_admin2', 'hour'])['total_trips'].mean().reset_index()
full_od_hf.columns = ['start_admin2', 'end_admin2', 'hour', 'total_trips_hf']
print(full_od_hf['total_trips'].sum())

full_od_te = full_od_n[full_od_n['timestamp'].dt.date == datetime.date(2021, 11, 24)]
full_od_te = full_od_te.groupby(['start_admin2', 'end_admin2', 'hour'])['total_trips'].mean().reset_index()
full_od_te.columns = ['start_admin2', 'end_admin2', 'hour', 'total_trips_te']
print(full_od_te['total_trips'].sum())

full_od_cd = full_od_n[full_od_n['timestamp'].dt.date == datetime.date(2020, 3, 21)]
full_od_cd = full_od_cd.groupby(['start_admin2', 'end_admin2', 'hour'])['total_trips'].mean().reset_index()
full_od_cd.columns = ['start_admin2', 'end_admin2', 'hour', 'total_trips_cd']
print(full_od_cd['total_trips'].sum())
# full_od_avg.groupby(['hour'])['total_trips'].mean().plot(marker='o')

# merge
full_ods = reduce(lambda left, right: pd.merge(left, right, on=['start_admin2', 'end_admin2', 'hour'], how='outer'),
                  [full_od_avg, full_od_ss, full_od_hf, full_od_te, full_od_cd])
full_ods = full_ods.fillna(0)
full_ods.to_pickle(r'D:\\NY_Emission\ODME_NY\OD_File\MDLD\od_event_ratio.pkl')

### 3. Get normal time series ratio from MDLD
data_path = r'E:\Dewey\Advan\\Neighborhood Patterns - US\\'
allfiles = glob.glob(data_path + '*.gz')
allfile = ([ff for ff in allfiles if '2023-02' in ff] + [ff for ff in allfiles if '2023-03' in ff] +
           [ff for ff in allfiles if '2023-04' in ff])
all_visits = pd.DataFrame()
for kk in tqdm(allfile):
    neighbor = pd.read_csv(kk)
    neighbor['BGFIPS'] = neighbor['AREA'].astype(str).apply(lambda x: x.zfill(12))
    neighbor = neighbor[neighbor['BGFIPS'].str[0:5].isin(CT_L)].reset_index(drop=True)

    # Transfer to Hourly visit
    date_range = [d.strftime('%Y-%m-%d %H:%M:%S') for d in
                  pd.date_range(neighbor.loc[0, 'DATE_RANGE_START'].split('T')[0],
                                neighbor.loc[0, 'DATE_RANGE_END'].split('T')[0], freq='h')][0: -1]
    day_visit = pd.DataFrame(neighbor['STOPS_BY_EACH_HOUR'].str[1:-1].str.split(',').tolist()).astype(int)
    day_visit.columns = date_range
    day_visit['BGFIPS'] = neighbor['BGFIPS'].str[0:-1]
    day_visit = day_visit.groupby(['BGFIPS']).sum().reset_index()
    day_visit = pd.melt(day_visit, id_vars=['BGFIPS'], value_vars=date_range)
    day_visit.columns = ['CTRFIPS', 'Timestamp', 'Visits']
    all_visits = pd.concat([all_visits, day_visit])
    all_visits = all_visits.groupby(['CTRFIPS', 'Timestamp']).sum().reset_index()

daily_ts = all_visits.groupby('Timestamp').sum()['Visits']
daily_ts.plot()
plt.show()
all_visits.to_pickle(r'D:\\NY_Emission\ODME_NY\OD_File\MDLD\Timeseries.pkl')

# Read total devices
# This is the monthly devices
devices = pd.read_csv(r'D:\NY_Emission\ODME_NY\OD_File\MDLD\device_count_cbg_ny_datea.csv', index_col=0)
ct_rp = {'CT.': '09', 'MA.': '25', 'NJ.': '34', 'NY.': '36', 'PA.': '42', 'US.': '', '\.': ''}
devices['BGFIPS'] = devices['home_block_group_id'].replace(ct_rp, regex=True)
# devices = devices.groupby('BGFIPS')['total_devices'].mean().reset_index()
CBG_features = pd.read_csv(r'F:\Research_Old\COVID19-Socio\Data\CBG_COVID_19.csv', index_col=0)
CBG_features['BGFIPS'] = CBG_features['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
devices = devices.merge(CBG_features, on='BGFIPS')
# devices['total_devices'] = devices['total_devices'] / 7
devices['weights'] = devices['Total_Population'] / devices['total_devices']
devices['penera'] = devices['total_devices'] / devices['Total_Population']
devices['MAPE'] = (devices['Total_Population'] - devices['total_devices']) / devices['Total_Population']
# devices = devices[(devices['weights'] < 30)&(devices['weights'] > 2)]
devices = devices[(devices['weights'] < 150)]
devices = devices[devices['BGFIPS'].str[0:5].isin(CT_NY)]
devices['weights'].describe()
devices['penera'].describe()
# set(devices['BGFIPS'].str[0:5])
sum(devices['total_devices']) / sum(devices['Total_Population'])
fig, ax = plt.subplots(figsize=(4.5, 4))
sns.regplot(data=devices, x='total_devices', y='Total_Population', ax=ax,
            label=r'$\rho = $' + str(round(devices[['total_devices', 'Total_Population']].corr().values[1][0], 3)),
            color='royalblue', scatter_kws={'alpha': 0.25})
# ax.plot([0, max(devices['Total_Population'])], [0, max(devices['Total_Population'])], '--', lw=3,
#         color='k')
plt.xlabel('Device count')
plt.ylabel('Total population')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\NY_Devices.pdf')
plt.savefig(r'D:\NY_Emission\Figure\NY_Devices.png', dpi=1000)

fig, ax = plt.subplots(figsize=(4.5, 4))
sns.distplot(devices['weights'], ax=ax)
plt.axvline(x=devices['weights'].mean(), color='k', linestyle='--')
plt.axvline(x=devices['weights'].median(), color='r', linestyle='--')
plt.xlabel('Population weight')
plt.tight_layout()
plt.legend(['Density', 'Mean', 'Median'])
plt.savefig(r'D:\NY_Emission\Figure\NY_Devices_dis.pdf')

# Trip rate
device_trips = pd.read_csv(r'D:\NY_Emission\ODME_NY\OD_File\MDLD\device_trip_cbg_ny.csv', index_col=0)
ct_rp = {'CT.': '09', 'MA.': '25', 'NJ.': '34', 'NY.': '36', 'PA.': '42', 'US.': '', '\.': ''}
device_trips['BGFIPS'] = device_trips['home_block_group_id'].replace(ct_rp, regex=True)
CBG_features = pd.read_csv(r'F:\Research_Old\COVID19-Socio\Data\CBG_COVID_19.csv', index_col=0)
CBG_features['BGFIPS'] = CBG_features['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
device_trips = device_trips.merge(CBG_features, on='BGFIPS')

fig, ax = plt.subplots(figsize=(4.5, 4))
sns.regplot(data=device_trips, x='trip_rate', y='Total_Population', ax=ax,
            label='$R^2 = $' + str(
                round(device_trips[['trip_rate', 'Total_Population']].corr().values[1][0], 3)),
            color='royalblue', scatter_kws={'alpha': 0.25})
ax.plot([0, max(devices['trip_rate'])], [0, max(devices['trip_rate'])], '--', lw=3,
        color='k')
plt.xlabel('Device count')
plt.ylabel('Total population')
plt.legend(loc='upper left')
plt.tight_layout()

fig, ax = plt.subplots(figsize=(4.5, 4))
sns.distplot(device_trips.loc[device_trips['trip_rate'] < 10, 'trip_rate'], ax=ax)
plt.axvline(x=device_trips['trip_rate'].mean(), color='k', linestyle='--')
plt.axvline(x=device_trips['trip_rate'].median(), color='r', linestyle='--')
plt.xlabel('Trip per person')
plt.tight_layout()
plt.legend(['density', 'mean', 'median'])
plt.savefig(r'D:\NY_Emission\Figure\NY_Devices_trips_dis.pdf')

