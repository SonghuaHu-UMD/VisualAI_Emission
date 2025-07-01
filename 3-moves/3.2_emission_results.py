import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import re
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d
from tqdm import tqdm
from scipy.optimize import curve_fit
import contextily as ctx
import mapclassify
import datetime
import matplotlib.dates as mdates
import matplotlib as mpl

pd.options.mode.chained_assignment = None
# Style for plot
plt.rcParams.update(
    {'font.size': 15, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

nmap = {1: 'TGH', 2: 'CO', 3: 'NOx', 87: 'VOC', 90: 'CO2', 91: 'TEC', 98: 'CO2E', 100: 'PM10', 110: 'PM2.5'}


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


#### 1. opmode change ####
no_signals = pd.DataFrame()
with_signals = pd.DataFrame()
for kk in range(0, 24):
    no_signal = pd.read_csv(r'D:\NY_Emission\MOVES\input_ns\median_opemode_%s.csv' % kk, index_col=0)
    no_signal = no_signal[['sourceTypeID', 'linkID', 'opModeID', 'opmodect']]
    no_signal.columns = ['sourceTypeID', 'linkID', 'opModeID', 'opmodect_no_signal']
    no_signal['hour'] = kk
    no_signals = pd.concat([no_signals, no_signal])
    with_signal = pd.read_csv(r'D:\NY_Emission\MOVES\input_\median_opemode_%s.csv' % kk, index_col=0)
    with_signal = with_signal[['sourceTypeID', 'linkID', 'opModeID', 'opmodect']]
    with_signal.columns = ['sourceTypeID', 'linkID', 'opModeID', 'opmodect_with_signal']
    with_signal['hour'] = kk
    with_signals = pd.concat([with_signals, with_signal])
signal_modes = no_signals.merge(with_signals, on=['sourceTypeID', 'linkID', 'opModeID', 'hour'], how='outer')
signal_modes = signal_modes.fillna(0)
signal_modes['diff'] = 100 * (signal_modes['opmodect_with_signal'] - signal_modes['opmodect_no_signal']) / 3600

osm_mt = pd.read_pickle(r'D:\NY_Emission\Shp\osmdta_ritis_signal.pkl')
osm_mt['linkID'] = osm_mt['from_node_'].astype(str) + '_' + osm_mt['to_node_id'].astype(str)
osm_mt = osm_mt.drop_duplicates(subset=['linkID']).reset_index(drop=True)
# osm_mt.drop('Cross_fclass', axis=1).to_file(r'D:\NY_Emission\Shp\osmdta_ritis_signal.shp')
signal_modes = signal_modes.merge(osm_mt[['linkID', 'Cross', 'Original', 'Is_signal']], on='linkID')

sns.set_palette(sns.color_palette('coolwarm', 3))
signal_modes = signal_modes[
    (signal_modes['Original'] != 'motorway') & (signal_modes['Is_signal'])]  # &(signal_modes['Is_signal'])
nmodes = signal_modes.groupby('opModeID')['diff'].mean().abs().sort_values().tail(10).index
for kk in range(0, 24):
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(x='diff', y='opModeID', hue='Original',
                data=signal_modes[(signal_modes['opModeID'].isin(nmodes)) & (signal_modes['hour'] == kk)],
                orient='h', ax=ax)
    ax.bar_label(ax.containers[0], fontsize=10, fmt='%.1f', padding=5)
    ax.bar_label(ax.containers[1], fontsize=10, fmt='%.1f', padding=5)
    ax.bar_label(ax.containers[2], fontsize=10, fmt='%.1f', padding=5)
    plt.legend(loc='lower left')
    plt.ylabel('Operation Mode')
    plt.xlabel('Difference (%)')
    plt.tight_layout()
    plt.savefig(r'D:\NY_Emission\Figure\Signal_Operation_mode_%s.png' % kk)
    plt.close()

fig, ax = plt.subplots(figsize=(5, 5))
sns.barplot(x='diff', y='opModeID', hue='Original',
            data=signal_modes[(signal_modes['opModeID'].isin(nmodes)) & (signal_modes['hour'] == 8)],
            orient='h', ax=ax)
plt.legend(loc='lower left')
plt.ylabel('Operation Mode')
plt.xlabel('Difference (%)')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\Signal_Operation_mode_8.pdf')
plt.close()

#### 2. age distribution ####
fig, ax = plt.subplots(figsize=(9, 5))
sns.set_palette('tab20')
ages = pd.read_csv(r'D:\NY_Emission\MOVES\input_\ageDistribution_2023.csv')
ages['sourceTypeID'] = ages['sourceTypeID'].astype(str)
sns.lineplot(y='ageFraction', x='ageID', hue='sourceTypeID', data=ages, orient='x')
plt.ylabel('Fraction')
plt.xlabel('Age (year)')
plt.legend(title='Source type', ncol=4)
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\Age_distri.pdf')

#### 3. opmode emission ####
emission_rate = pd.read_csv(r'D:\NY_Emission\MOVES\MatrixData\newyork\f2i33_2023_1_40_55.csv')
emission_rate.columns = ['opModeID', 'pollutantID', 'sourceTypeID', 'modelYearID', 'em', 'hehe']
emission_rate = emission_rate.replace({'pollutantID': nmap})
emission_rate = emission_rate[emission_rate['pollutantID'].isin(['CO', 'NOx', 'CO2', 'PM10', 'PM2.5'])]
# for stt in [21, 31, 52, 61, 42, 43]:
for pid in ['CO', 'NOx', 'CO2', 'PM10', 'PM2.5']:
    emission_raten = emission_rate[
        (emission_rate['sourceTypeID'].isin([21, 31, 42, 43, 52, 61, ])) & (emission_rate['pollutantID'] == pid) & (
            emission_rate['modelYearID'].isin([2020]))]  # 2000, 2005, 2010, 2015, 2020
    emission_raten['em'] = emission_raten['em'] / 3600
    emission_raten = emission_raten[emission_raten['opModeID'] != 300]
    fig, ax = plt.subplots(figsize=(5, 6))
    # sns.set_palette(sns.color_palette('coolwarm', 5))
    sns.barplot(y='opModeID', x='em', hue='sourceTypeID', data=emission_raten, orient='h',
                palette='Set2')
    plt.xlabel('Emission rate (g/s)')
    plt.ylabel('Operating mode')
    plt.legend(title='Source type')
    plt.tight_layout()
    plt.savefig(r'D:\NY_Emission\Figure\FEmission_rate_%s.pdf' % (pid))
    plt.close()

#### 4. ablation analysis ####
# Read emission: base
all_files_inrix = glob.glob(r'D:\NY_Emission\MOVES\output_inrix\*_emissionbylink.csv')
emission_inrix = pd.DataFrame()
for kk in all_files_inrix:
    df = pd.read_csv(kk)
    df['Hour'] = re.findall(r'\d+', kk)[0]
    df['emquant'] = df['emquant'] * 0.000621371  # meter to miles since emrate is g/mile/hour
    emission_inrix = pd.concat([emission_inrix, df])
emission_inrix['Hour'] = emission_inrix['Hour'].astype(int)
emission_inrix.loc[emission_inrix['Hour'] > 23, 'Hour'] = (emission_inrix.loc[emission_inrix['Hour'] > 23, 'Hour'] - 24)
emission_inrix = emission_inrix.sort_values(by=['linkID', 'pollutantID', 'Hour'])
emission_inrix = emission_inrix.replace({'pollutantID': nmap})
emission_inrix.columns = ['linkID', 'pollutantID', 'emrate_inrix', 'emquant_inrix', 'Hour']

# Read emission: with signal
all_files_signal = glob.glob(r'D:\NY_Emission\MOVES\output_signal\*_emissionbylink.csv')
emission_signal = pd.DataFrame()
for kk in all_files_signal:
    df = pd.read_csv(kk)
    df['Hour'] = re.findall(r'\d+', kk)[0]
    df['emquant'] = df['emquant'] * 0.000621371  # meter to mile since emrate is g/mile/vehicle
    emission_signal = pd.concat([emission_signal, df])
emission_signal['Hour'] = emission_signal['Hour'].astype(int)
emission_signal = emission_signal.sort_values(by=['linkID', 'pollutantID', 'Hour'])
emission_signal = emission_signal.replace({'pollutantID': nmap})
emission_signal.columns = ['linkID', 'pollutantID', 'emrate_signal', 'emquant_signal', 'Hour']

# Merge
emi_all = emission_signal.merge(emission_inrix, on=['linkID', 'pollutantID', 'Hour'], how='inner')
osm_mt = pd.read_pickle(r'D:\NY_Emission\Shp\osmdta_ritis_signal.pkl')
osm_mt['linkID'] = osm_mt['from_node_'].astype(str) + '_' + osm_mt['to_node_id'].astype(str)
osm_mt = osm_mt.drop_duplicates(subset=['linkID']).reset_index(drop=True)
# osm_mt.drop('Cross_fclass', axis=1).to_file(r'D:\NY_Emission\Shp\osmdta_ritis_signal.shp')
emi_all = emi_all.merge(osm_mt[['linkID', 'Cross', 'Original', 'Is_signal']], on='linkID')

# Get need pollutants
emi_all = emi_all[emi_all['pollutantID'].isin(['CO', 'NOx', 'CO2', 'PM2.5'])]

# Consider spatiotemporal change
emi_all['emquant_signal_mean'] = emi_all.groupby(['linkID', 'pollutantID'])['emquant_signal'].transform("mean")
emi_all['emquant_signal_diff'] = 100 * (emi_all['emquant_signal'] - emi_all['emquant_signal_mean']) / \
                                 emi_all['emquant_signal_mean']

# Plot hourly change
sns.set_palette(sns.color_palette('coolwarm', 4))
yy = 'emquant_signal_diff'
# fig, axs = plt.subplots(figsize=(12, 5))
fig, axs = plt.subplots(figsize=(6, 4))
sns.boxplot(emi_all, x='Hour', y=yy, hue='pollutantID', showmeans=True, meanline=True,
            meanprops=dict(linestyle='-', linewidth=1, color='purple'), showfliers=False,
            flierprops=dict(marker='o', markerfacecolor='None', markersize=1, linestyle='none', markeredgecolor='gray',
                            alpha=0.2), )
plt.ylim(min(emi_all[yy]) * 1.1, max(emi_all[yy]) / 4)
plt.legend(title='', loc='upper left')
plt.ylabel('Emission variation (%)', labelpad=-10)
plt.xticks(np.arange(0, 24, 3))
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
# plt.subplots_adjust(top=0.967, bottom=0.112, left=0.13, right=0.986, hspace=0.2, wspace=0.2)
plt.savefig(r'D:\NY_Emission\Figure\pollutant_time_diff_s.pdf')
plt.close()

# Comparison 1: Signal (emquant_) vs no-signal (emquant_ns)
# Calculate the probability of traffic which meets the red light
emi_all['emquant_sn'] = emi_all['emquant_signal'] * emi_all['Rt'] + emi_all['emquant_inrix'] * (1 - emi_all['Rt'])
emi_all['diff'] = emi_all['emquant_inrix'] - emi_all['emquant_sn']
emi_all['pdiff'] = 100 * emi_all['diff'] / emi_all['emquant_sn']

sns.set_palette(sns.color_palette('coolwarm', 4))
yy = 'pdiff'
emi_all_pl = emi_all.copy()
fig, axs = plt.subplots(figsize=(6, 4.5))
sns.boxplot(emi_all_pl, x=yy, y='Original', hue='pollutantID', showmeans=True, meanline=True,
            meanprops=dict(linestyle='-', linewidth=1, color='purple'), showfliers=False,
            flierprops=dict(marker='o', markerfacecolor='None', markersize=1, linestyle='none', markeredgecolor='gray',
                            alpha=0.2), )
# plt.xlim(min(emi_all[yy].dropna()), max(emi_all[yy].dropna()) / 1)
plt.legend(title='', loc='lower right', ncol=1)
plt.xlabel('Emission difference (%)')
plt.ylabel('')
plt.axvline(x=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\pollutant_aba_singal_diff.pdf')
plt.close()

# Comparison 2: Normal (emquant_) vs Abnormal (emquant_ns)
cct = 0
# Four events: snow storm (2022/1/29); covid (2020/3/21); Thanksgiving Eve (2021/11/24); Henri flooding (2021/8/22)
for rr in ['cd', 'te', 'hf', 'ss', 'nv', 'ns', '', 'nsp', 'nvo', 'ncamera']:
    all_files_inrix = glob.glob(r'D:\NY_Emission\MOVES\input_%s\Output\*_emissionbylink.csv' % rr)
    emission_inrix = pd.DataFrame()
    if len(all_files_inrix) > 0:
        for kk in all_files_inrix:
            df = pd.read_csv(kk)
            t_hour = re.findall(r'\d+', kk)[0]
            df['Hour'] = t_hour
            df['emquant'] = df['emquant'] * 0.000621371  # meter to miles since emrate is g/mile/hour
            tvs = pd.read_csv((r'D:\NY_Emission\MOVES\input_%s\link_NYC_36061_%s.csv' % (rr, t_hour)))
            df = df.merge(tvs[['linkID', 'linkLength', 'linkVolume', 'linkAvgSpeed']], on='linkID')
            emission_inrix = pd.concat([emission_inrix, df])
        emission_inrix['Hour'] = emission_inrix['Hour'].astype(int)
        emission_inrix = emission_inrix.sort_values(by=['linkID', 'pollutantID', 'Hour'])
        emission_inrix = emission_inrix.replace({'pollutantID': nmap})
        emission_inrix.columns = ['linkID', 'pollutantID', 'emrate_%s' % rr, 'emquant_%s' % rr, 'Hour',
                                  'linkLength_%s' % rr, 'linkVolume_%s' % rr, 'linkAvgSpeed_%s' % rr]
        if cct == 0:
            emi_all = emission_inrix
        else:
            emi_all = emi_all.merge(emission_inrix, on=['linkID', 'pollutantID', 'Hour'], how='inner')
        cct += 1

# Get need pollutants
# emi_all_us = emi_all.copy()
emi_all = emi_all[emi_all['pollutantID'].isin(['CO', 'NOx', 'CO2', 'PM2.5'])]
emi_dayp = emi_all.groupby(['pollutantID'])[['emquant_', 'emquant_cd', 'emquant_te', 'emquant_hf', 'emquant_ss']].sum()

linkids = ['205394_212547', '139_199908', '36991_70', '37461_37462', '36987_37451', '127952_205838', '127909_83',
           '8253_5756', '239542_242107', '204866_204867', '250349_250350', '201655_253985', '201652_201653',
           '260635_260633', '124250_124251', '199490_199502', '199488_224461', '683_684', '205615_37143',
           '237147_237148', '199563_199564']
# Plot spatial map: by pollutants
plot_y = 'emrate_'
for kk in ['CO', 'NOx', 'CO2', 'PM2.5']:
    # hour_in = 8
    emi_all_need = emi_all[(emi_all['pollutantID'] == kk)]
    if kk in ['CO', 'NOx']:
        emi_all_need[plot_y] = 1.2 * emi_all_need[plot_y] / 1000  # to kg
    if kk == 'CO2':
        emi_all_need[plot_y] = 1.2 * emi_all_need[plot_y] / 1e6  # to ton
    if kk == 'PM2.5':
        emi_all_need[plot_y] = (1.2 * 1.5 * emi_all_need[plot_y]) / 1e3  # to kg
    binning = mapclassify.NaturalBreaks(emi_all_need[plot_y], k=5)  # NaturalBreaks
    emi_all_need['cut_jenks'] = (binning.yb + 1)
    for hour_in in [3, 8, 16]:
        # Plot spatial dynamics
        osm_mt_am = osm_mt.merge(emi_all_need[emi_all_need['Hour'] == hour_in], on='linkID', how='left').reset_index(
            drop=True)
        osm_mt_am = osm_mt_am.replace([np.inf, -np.inf], np.nan)
        osm_mt_am = osm_mt_am.fillna(0)
        # osm_mt_am[['linkID','emrate_','geometry','emquant_']].to_file(r'D:\NY_Emission\Shp\osm_mt_am_emrate.shp')
        AvgErate = osm_mt_am['emrate_'].mean()
        BridErate = osm_mt_am[osm_mt_am['linkID'].isin(linkids)]['emrate_'].mean()
        print('Hour: %s Polluant %s AvgErate: %s BridErate: %s Pct: %s' % (
            hour_in, kk, AvgErate, BridErate, (BridErate - AvgErate) / AvgErate))
        osm_mt_am = osm_mt_am.to_crs('EPSG:32618')
        fig, ax = plt.subplots(figsize=(3.5, 7))
        mpl.rcParams['text.color'] = 'w'
        osm_mt_am.plot(column=plot_y, cmap='RdYlGn_r', scheme="user_defined",
                       classification_kwds={'bins': binning.bins},
                       lw=osm_mt_am['cut_jenks'], ax=ax, alpha=0.6, legend=True,
                       legend_kwds={'labelcolor': 'white', "fmt": "{:.2f}", 'ncol': 1, 'title': kk,
                                    'loc': 'upper left', 'frameon': True, 'facecolor': 'k', 'edgecolor': 'k',
                                    'framealpha': 0.5})
        mpl.rcParams['text.color'] = 'k'
        ctx.add_basemap(ax, crs=osm_mt_am.crs, source=ctx.providers.CartoDB.DarkMatter, alpha=0.9)
        plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
        # plt.tight_layout()
        plt.axis('off')
        plt.savefig(r'D:\NY_Emission\Figure\pollutant_spatial_distribution_%s_%s.pdf' % (kk, hour_in))
        plt.close()

# Merge with road features
osm_mt = pd.read_pickle(r'D:\NY_Emission\Shp\osmdta_ritis_signal.pkl')
osm_mt['linkID'] = osm_mt['from_node_'].astype(str) + '_' + osm_mt['to_node_id'].astype(str)
osm_mt = osm_mt.drop_duplicates(subset=['linkID']).reset_index(drop=True)
# osm_mt.drop('Cross_fclass', axis=1).to_file(r'D:\NY_Emission\Shp\osmdta_ritis_signal.shp')
emi_all = emi_all.merge(osm_mt[['linkID', 'Cross', 'Original', 'Is_signal']], on='linkID')

# Check emission variation near bridge density
AvgErate = emi_all[~emi_all['linkID'].isin(linkids)].groupby(['pollutantID', 'Hour'])['emrate_'].mean().reset_index()
BridErate = emi_all[emi_all['linkID'].isin(linkids)].groupby(['pollutantID', 'Hour'])['emrate_'].mean().reset_index()
AvgErate = AvgErate.merge(BridErate, on=['pollutantID', 'Hour'], how='left')
AvgErate['diff'] = (AvgErate['emrate__y'] - AvgErate['emrate__x']) / AvgErate['emrate__x']
AvgErate.groupby('pollutantID')['diff'].mean()
AvgErate[AvgErate['Hour'] == 17]

# Check total emission variation by road types
emi_all.groupby('Original')['emquant_'].sum() / emi_all['emquant_'].sum()
emi_all.groupby('Original')['linkLength_'].sum() / emi_all['linkLength_'].sum()

# Check emission rate variation by roads
emi_all_sum = emi_all.groupby(["linkID", 'pollutantID', "Original"])["emrate_"].mean().reset_index()
emi_all_sum['avg_emrate_'] = emi_all_sum.groupby(["Original", 'pollutantID'])["emrate_"].transform('mean')
emi_all_sum['emrate_diff'] = (emi_all_sum['emrate_'] - emi_all_sum['avg_emrate_']) / emi_all_sum['avg_emrate_']
emi_all_sum.groupby(['pollutantID', 'Original'])['emrate_diff'].quantile(0.95)
emi_all_sum.groupby(['pollutantID', 'Original'])['emrate_diff'].quantile(0.05)

# Check emission by hour
hour_var = (emi_all.groupby(['pollutantID', 'Hour'])['emquant_'].sum() / emi_all.groupby(['pollutantID'])[
    'emquant_'].sum()).reset_index()
hour_var[hour_var['Hour'].isin([7, 8, 9, 16, 17, 18])].groupby('pollutantID').sum()['emquant_']

hour_var = (emi_all.groupby(['pollutantID', 'Hour'])['emquant_'].mean() / emi_all.groupby(['pollutantID'])[
    'emquant_'].mean()).reset_index()

# consider volume before ODME
assign_all_before = pd.read_pickle(r'D:\NY_Emission\ODME_NY\Simulation_outcome\assign_all_before_.pkl')
assign_all_before['linkID'] = assign_all_before['from_node_id'].astype(str) + '_' + assign_all_before[
    'to_node_id'].astype(str)
assign_all_before = assign_all_before[['linkID', 'hour', 'volume_hourly']]
assign_all_before.columns = ['linkID', 'Hour', 'volume_hourly_before']
emi_all = emi_all.merge(assign_all_before, on=['linkID', 'Hour'], how='left')
# plt.plot(emi_all['volume_hourly_before'],emi_all['linkVolume_'],'o')
emi_all['emissions_'] = emi_all['emrate_'] * emi_all['linkAvgSpeed_'] / emi_all['linkVolume_']
emi_all['emquant_nvo1'] = (0.000621371 * emi_all['emissions_'] * emi_all['linkLength_'] * emi_all[
    'volume_hourly_before'] / emi_all['linkAvgSpeed_']) * (
                                  sum(emi_all['linkVolume_']) / sum(emi_all['volume_hourly_before']))
# (sum(emi_all['volume_hourly_before'])-sum(emi_all['linkVolume_']))/sum(emi_all['volume_hourly_before'])
emi_all['emquant_ncamera1'] = emi_all['emquant_ncamera'] * (
            emi_all['volume_hourly_before'] / emi_all['linkVolume_ncamera'])

for rr in ['cd', 'te', 'hf', 'ss', 'nv', 'ns', 'nsp', 'nvo', 'ncamera1']:  # 'ss', 'nv'
    emi_all['diff_VS%s' % rr] = emi_all['emquant_%s' % rr] - emi_all['emquant_']
    emi_all['pdiff_VS%s' % rr] = 100 * emi_all['diff_VS%s' % rr] / (emi_all['emquant_'])
    emission_s = emi_all.drop(['linkID'], axis=1).groupby(['Hour', 'pollutantID', 'Original']).sum().reset_index()
    emission_s['pdiff_VS%s' % rr] = 100 * emission_s['diff_VS%s' % rr] / emission_s['emquant_']
    # Total
    emission_st = emission_s.groupby(['pollutantID']).sum().reset_index()
    emission_st['pdiff_VS%s' % rr] = 100 * emission_st['diff_VS%s' % rr] / emission_st['emquant_']
    # sns.barplot(emission_st, x='Hour', y='pdiff_VS%s' % rr, hue='pollutantID')
    print(emission_st['pdiff_VS%s' % rr])
    # print('Case %s decreases %s' % (rr, 100 * emission_st['diff_VS%s' % rr].sum() / emission_st['emquant_'].sum()))

sns.set_palette(sns.color_palette('coolwarm', 4))
# temp=emi_all[['linkVolume_','avg_volume','linkVolume_nvo','pdiff_VSnvo1']]
for rr in ['nv', 'nsp', 'ncamera1']:
    yy = 'pdiff_VS%s' % rr
    fig, axs = plt.subplots(figsize=(6, 4.5))
    emi_alln = emi_all[emi_all['pdiff_VS%s' % rr].abs() < 250]
    print(rr + '--------------------')
    print(100 * (emi_alln.groupby('pollutantID')['emquant_'].sum() - emi_alln.groupby('pollutantID')[
        'emquant_%s' % rr].sum()) / emi_alln.groupby('pollutantID')['emquant_'].sum())
    print(emi_alln.groupby(['pollutantID'])['pdiff_VS%s' % rr].quantile([0.5]))

    sns.boxplot(emi_alln, x=yy, y='Original', hue='pollutantID', showmeans=True, meanline=True,
                meanprops=dict(linestyle='-', linewidth=1, color='purple'), showfliers=False,
                flierprops=dict(marker='o', markerfacecolor='None', markersize=1, linestyle='none',
                                markeredgecolor='gray', alpha=0.2), )
    # plt.xlim(min(emi_all[yy].dropna()) * 1.1, max(emi_all[yy].dropna()) / 1.2)
    plt.legend(title='', loc='lower right', ncol=1)
    plt.xlabel('Emission difference (%)')
    plt.ylabel('')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(r'D:\NY_Emission\Figure\pollutant_aba_%s_diff.pdf' % rr)
    plt.close()

# Plot spatial dynamics
osm_mt_am = osm_mt.merge(emi_all[(emi_all['Hour'] == 8) & (emi_all['pollutantID'] == 'CO2')], on='linkID').reset_index(
    drop=True)
osm_mt_am = osm_mt_am.replace([np.inf, -np.inf], np.nan)
osm_mt_am = osm_mt_am.fillna(0)
for rr in ['cd', 'te', 'hf', 'ss']:
    binning = mapclassify.NaturalBreaks(osm_mt_am['pdiff_VS%s' % rr], k=5)  # NaturalBreaks
    osm_mt_am['cut_jenks'] = (binning.yb + 1) * 0.5
    osm_mt_am = osm_mt_am.to_crs('EPSG:32618')
    fig, ax = plt.subplots(figsize=(3.5, 7))
    mpl.rcParams['text.color'] = 'w'  # osm_mt_am['cut_jenks']
    osm_mt_am.plot(column='pdiff_VS%s' % rr, cmap='RdYlGn_r', scheme="user_defined",
                   classification_kwds={'bins': [-75, -50, 0, 50, 75]},
                   lw=1, ax=ax, alpha=0.6, legend=True,
                   legend_kwds={'labelcolor': 'white', "fmt": "{:.0f}", 'ncol': 1, 'title': 'CO2 Change (%)',
                                'loc': 'upper left', 'frameon': True, 'facecolor': 'k', 'edgecolor': 'k',
                                'framealpha': 0.5})
    mpl.rcParams['text.color'] = 'k'
    ctx.add_basemap(ax, crs=osm_mt_am.crs, source=ctx.providers.CartoDB.DarkMatter, alpha=0.9)
    plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
    plt.axis('off')
    plt.savefig(r'D:\NY_Emission\Figure\pdiff_VS%s_spatial_change.pdf' % rr)
    plt.close()

# Get emissions by sourcetype
nmap = {1: 'TGH', 2: 'CO', 3: 'NOx', 87: 'VOC', 90: 'CO2', 91: 'TEC', 98: 'CO2E', 100: 'PM10', 110: 'PM2.5'}
# Read emission: base
all_files_source = glob.glob(r'D:\NY_Emission\MOVES\input_\Output\*_emissionbylinksource.csv')
emission_source = pd.DataFrame()
for kk in all_files_source:
    df = pd.read_csv(kk)
    df['Hour'] = re.findall(r'\d+', kk)[0]
    df['emquant'] = df['emquant'] * 0.000621371  # meter to miles
    emission_source = pd.concat([emission_source, df])
emission_source['Hour'] = emission_source['Hour'].astype(int)
emission_source = emission_source.replace({'pollutantID': nmap})
# Get need pollutants
emission_source = emission_source[emission_source['pollutantID'].isin(['CO', 'NOx', 'CO2', 'PM2.5'])]
emission_source_avg = (emission_source.groupby(['sourceTypeID', 'pollutantID'])['emquant'].sum() / \
                       emission_source.groupby(['pollutantID'])['emquant'].sum()).reset_index()
emission_source_avg = emission_source_avg.sort_values(by=['pollutantID', 'sourceTypeID'])
emission_source_avg['sourceTypeID'] = emission_source_avg['sourceTypeID'].astype(str)
emission_source_avg = emission_source_avg[emission_source_avg['emquant'] > 0]

smap = {'11': 'Motorcycle', '21': 'Passenger Car', '31': 'Passenger Truck', '42': 'Transit Bus', '43': 'School Bus',
        '52': 'Single-Unit Truck', '61': 'Combination Truck'}
emission_source_avg['sourceTypeID'] = emission_source_avg['sourceTypeID'].replace(smap)
emission_source_avg = emission_source_avg[emission_source_avg['sourceTypeID'] != 'Motorcycle']
emission_source_avg['emquant'] = emission_source_avg['emquant'] * 100

# Plot emission by sourcetype
sns.set_palette('Set2')
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=emission_source_avg, x="pollutantID", hue="sourceTypeID", weights='emquant', multiple="stack",
             discrete=True, shrink=.8)
sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, 1.46), ncol=2, title='')
plt.ylabel('Percentage(%)')
plt.xlabel('Emission type')
plt.tight_layout()
plt.subplots_adjust(top=0.75, bottom=0.15)
plt.savefig(r'D:\NY_Emission\Figure\emission_sourcetype.pdf')

# mobility for four events
sum_day_ny = pd.read_csv(r'D:\\NY_Emission\ODME_NY\OD_File\MDLD\sum_day_ny.csv')
mean_2021 = sum_day_ny.loc[sum_day_ny['date'].dt.year == 2021, 'emission_f'].mean()
sum_day_ny['emission_diff'] = (sum_day_ny['emission_f'] - mean_2021) / mean_2021
sum_day_ny['emission_diff_all'] = (sum_day_ny['emission_f'] - sum_day_ny['emission_f'].mean()) / sum_day_ny[
    'emission_f'].mean()
sum_day_ny['emission_diff'].describe()
# snow storm (2021/2/1); covid (2020/3/21); Thanksgiving Eve (2021/11/26); Henri flooding (2021/8/22)
sns.set_palette(sns.color_palette('coolwarm', 4))
ess = ['COVID-19 \nPandemic', 'Black \nFriday', 'Henri \nFlooding', 'Snow \nStorm']
dts = [datetime.datetime(2020, 3, 22, ), datetime.datetime(2021, 11, 26),
       datetime.datetime(2021, 8, 22), datetime.datetime(2021, 2, 1)]
sum_day_ny[sum_day_ny['date'].isin(dts)]
(sum_day_ny[sum_day_ny['date'].dt.year == 2020]['emission_f'].sum() - sum_day_ny[sum_day_ny['date'].dt.year == 2021][
    'emission_f'].sum()) / sum_day_ny[sum_day_ny['date'].dt.year == 2021]['emission_f'].sum()
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sum_day_ny['date'], sum_day_ny['emission_f'])
for kk in range(0, 4):
    need_p = sum_day_ny[sum_day_ny['date'] == dts[kk]]
    ax.axvline(x=dts[kk], color='r', ymin=0, ymax=1, linestyle='--', alpha=0.5)
    ax.plot(need_p['date'], need_p['emission_f'], marker="o", markerfacecolor="None",
            markeredgecolor='red')
    if kk == 0:
        kss = 0
    elif kk == 1:
        kss = 1
    elif kk == 2:
        kss = kk + 3
    elif kk == 3:
        kss = kk + 4
    ax.annotate(ess[kk], xy=(dts[kk], max(sum_day_ny['emission_f']) * (0.9 - kss * 0.1)),
                xytext=(dts[kk] - datetime.timedelta(days=100), max(sum_day_ny['emission_f']) * (0.9 - kss * 0.1)),
                color='k')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=8))
plt.ylabel('CO2 Emission (tons)')
plt.xlabel('Date')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.subplots_adjust(top=0.937, bottom=0.112, left=0.15, right=0.976, hspace=0.2, wspace=0.2)
# plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\emission_daily.pdf')
