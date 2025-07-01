import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import re
import seaborn as sns
import contextily as ctx
import mapclassify
import matplotlib as mpl
import geopandas as gpd

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

cct = 0
for rr in ['s_mode10', 's_mode20', 's_mode30', 's_peak10', 's_peak20', 's_peak30', 's_remote10', 's_remote20',
           's_remote30', 's_raw']:
    all_files_inrix = glob.glob(r'D:\NY_Emission\MOVES\input_%s\Output\*_emissionbylink.csv' % rr)
    emission_inrix = pd.DataFrame()
    if len(all_files_inrix) > 0:
        for kk in all_files_inrix:
            df = pd.read_csv(kk)
            t_hour = re.findall(r'\d+', kk)
            if len(t_hour) > 1:
                t_hour = t_hour[1]
            else:
                t_hour = t_hour[0]
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

# build electric
all_files_inrix = glob.glob(r'D:\NY_Emission\MOVES\input_s_raw\Output\*_emissionbylinksource.csv')
emission_source = pd.DataFrame()
if len(all_files_inrix) > 0:
    for kk in all_files_inrix:
        df = pd.read_csv(kk)
        t_hour = re.findall(r'\d+', kk)[0]
        df['Hour'] = t_hour
        df['emquant'] = df['emquant'] * 0.000621371  # meter to miles since emrate is g/mile/hour
        emission_source = pd.concat([emission_source, df])
    emission_source['Hour'] = emission_source['Hour'].astype(int)
    emission_source = emission_source.sort_values(by=['linkID', 'pollutantID', 'Hour'])
    emission_source = emission_source.replace({'pollutantID': nmap})

# electric by vehicle type
emission_source = emission_source[emission_source['pollutantID'].isin(['CO', 'NOx', 'CO2', 'PM2.5'])]
emission_source = emission_source.groupby(['linkID', 'sourceTypeID', 'pollutantID'])['emquant'].sum().reset_index()
smap = {'11': 'Motorcycle', '21': 'Passenger Car', '31': 'Passenger Truck', '42': 'Transit Bus', '43': 'School Bus',
        '52': 'Single-Unit Truck', '61': 'Combination Truck'}
emission_source['sourceTypeID'] = emission_source['sourceTypeID'].astype(str)
emission_source['sourceTypeID'] = emission_source['sourceTypeID'].replace(smap)

cct = 0
for kk in ['Passenger Car', 'Passenger Truck', 'Transit Bus', 'School Bus', 'Single-Unit Truck', 'Combination Truck']:
    emission_source_ele = emission_source[emission_source['sourceTypeID'] != kk]
    emission_source_ele = emission_source_ele.groupby(['linkID', 'pollutantID'])['emquant'].sum().reset_index()
    emission_source_ele.columns = ['linkID', 'pollutantID', 'emquant_%s' % kk]
    if cct == 0:
        emission_eles = emission_source_ele
    else:
        emission_eles = emission_eles.merge(emission_source_ele, on=['linkID', 'pollutantID'], how='inner')
    cct += 1

# Get need pollutants
all_var = ['emquant_s_mode10', 'emquant_s_mode20', 'emquant_s_mode30', 'emquant_s_peak10', 'emquant_s_peak20',
           'emquant_s_peak30', 'emquant_s_remote10', 'emquant_s_remote20', 'emquant_s_remote30']
emi_all = emi_all[emi_all['pollutantID'].isin(['CO', 'NOx', 'CO2', 'PM2.5'])]
emi_all = emi_all.groupby(['linkID', 'pollutantID'])[all_var + ['emquant_s_raw']].sum().reset_index()
emi_all = emi_all.merge(emission_eles, on=['linkID', 'pollutantID'], how='inner')
emi_dayp = emi_all.groupby(['pollutantID'])[all_var + ['emquant_s_raw'] + list(emission_eles.columns)[2:]].sum()

# pct
for rr in all_var + list(emission_eles.columns)[2:]:
    emi_dayp[rr] = (emi_dayp[rr] - emi_dayp['emquant_s_raw']) / emi_dayp['emquant_s_raw']
    emi_all[rr + '_pct'] = 100 * (emi_all[rr] - emi_all['emquant_s_raw']) / emi_all['emquant_s_raw']

# mode
df_mode = pd.melt(
    emi_all[['linkID', 'pollutantID', 'emquant_s_mode10_pct', 'emquant_s_mode20_pct', 'emquant_s_mode30_pct']],
    id_vars=['linkID', 'pollutantID'],
    value_vars=['emquant_s_mode10_pct', 'emquant_s_mode20_pct', 'emquant_s_mode30_pct'])
print(df_mode.groupby(['variable', 'pollutantID'])['value'].mean())
custom_dict = {'CO': 0, 'CO2': 2, 'NOx': 1, 'PM2.5': 3}
df_mode = df_mode.sort_values(by=['linkID', 'pollutantID', 'variable'], key=lambda x: x.map(custom_dict))
sns.set_palette(sns.color_palette('coolwarm', 4))
fig, ax = plt.subplots(figsize=(6.5, 5))
sns.boxplot(df_mode, x='variable', y='value', hue='pollutantID', showfliers=False, showmeans=True,
            meanline=True, meanprops=dict(linestyle='-', linewidth=1, color='purple'), ax=ax)
ax.set_xticklabels([10, 20, 30])
plt.xlabel('Mode shift (%)')
plt.ylabel('Emission change (%)')
plt.legend(title='')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\E_by_mode_shift.pdf')


# peak
df_peak = pd.melt(
    emi_all[['linkID', 'pollutantID', 'emquant_s_peak10_pct', 'emquant_s_peak20_pct', 'emquant_s_peak30_pct']],
    id_vars=['linkID', 'pollutantID'],
    value_vars=['emquant_s_peak10_pct', 'emquant_s_peak20_pct', 'emquant_s_peak30_pct'])
print(df_peak.groupby(['variable', 'pollutantID'])['value'].mean())
custom_dict = {'CO': 0, 'CO2': 2, 'NOx': 1, 'PM2.5': 3}
df_peak = df_peak.sort_values(by=['linkID', 'pollutantID', 'variable'], key=lambda x: x.map(custom_dict))
sns.set_palette(sns.color_palette('coolwarm', 4))
fig, ax = plt.subplots(figsize=(6.5, 5))
sns.boxplot(df_peak, x='variable', y='value', hue='pollutantID', showfliers=False,
            showmeans=True, meanline=True, meanprops=dict(linestyle='-', linewidth=1, color='purple'), ax=ax)
ax.set_xticklabels([10, 20, 30])
plt.xlabel('Departure time shift (%)')
plt.ylabel('Emission change (%)')
plt.legend(title='')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\E_peak_shift.pdf')

# ev
df_ev = pd.melt(
    emi_all[['linkID', 'pollutantID', 'emquant_Passenger Car_pct', 'emquant_Passenger Truck_pct',
             'emquant_Transit Bus_pct', 'emquant_School Bus_pct', 'emquant_Single-Unit Truck_pct',
             'emquant_Combination Truck_pct']],
    id_vars=['linkID', 'pollutantID'],
    value_vars=['emquant_Passenger Car_pct', 'emquant_Passenger Truck_pct', 'emquant_Transit Bus_pct',
                'emquant_School Bus_pct', 'emquant_Single-Unit Truck_pct', 'emquant_Combination Truck_pct'])
custom_dict = {'CO': 0, 'CO2': 2, 'NOx': 1, 'PM2.5': 3}
df_ev = df_ev.sort_values(by=['linkID', 'pollutantID', 'variable'], key=lambda x: x.map(custom_dict))
print(df_ev.groupby(['variable', 'pollutantID'])['value'].mean())
sns.set_palette(sns.color_palette('coolwarm', 4))
fig, ax = plt.subplots(figsize=(6.5, 5))
sns.boxplot(df_ev, x='variable', y='value', hue='pollutantID', showfliers=False,
            showmeans=True, meanline=True, meanprops=dict(linestyle='-', linewidth=1, color='purple'), ax=ax)
ax.set_xticklabels(['P_Car', 'P_Truck', 'T_Bus', 'S_Bus', 'SU_Truck', 'C_Truck'])
plt.xlabel('Electrification')
plt.ylabel('Emission change (%)')
plt.xticks(rotation=15)
plt.legend(title='', loc='lower right')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\E_ev_shift.pdf')

# congestion pricing
cct = 0
for rr in ['s_cong_2', 's_cong_4', 's_cong_6', 's_cong_8', 's_raw']:
    all_files_inrix = glob.glob(r'D:\NY_Emission\MOVES\input_%s\Output\*_emissionbylink.csv' % rr)
    emission_inrix = pd.DataFrame()
    if len(all_files_inrix) > 0:
        for kk in all_files_inrix:
            df = pd.read_csv(kk)
            t_hour = re.findall(r'\d+', kk)
            if len(t_hour) > 1:
                t_hour = t_hour[1]
            else:
                t_hour = t_hour[0]
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

# Plot spatial dynamics
osm_mt = pd.read_pickle(r'D:\NY_Emission\Shp\osmdta_ritis_signal.pkl')
osm_mt['linkID'] = osm_mt['from_node_'].astype(str) + '_' + osm_mt['to_node_id'].astype(str)
osm_mt = osm_mt.drop_duplicates(subset=['linkID']).reset_index(drop=True)
# osm_mt.drop('Cross_fclass', axis=1).to_file(r'D:\NY_Emission\Shp\osmdta_ritis_signal.shp')
osm_mt_am = osm_mt.merge(emi_all[(emi_all['Hour'] == 8) & (emi_all['pollutantID'] == 'CO2')], on='linkID').reset_index(
    drop=True)

# Get need pollutants
all_var = ['emquant_s_cong_2', 'emquant_s_cong_4', 'emquant_s_cong_6', 'emquant_s_cong_8']
emi_all = emi_all[emi_all['pollutantID'].isin(['CO', 'NOx', 'CO2', 'PM2.5'])]
emi_all = emi_all.groupby(['linkID', 'pollutantID'])[all_var + ['emquant_s_raw']].sum().reset_index()
# pct
for rr in all_var:
    emi_all[rr + '_pct'] = 100 * (emi_all[rr] - emi_all['emquant_s_raw']) / emi_all['emquant_s_raw']

# plot
df_cong = pd.melt(
    emi_all[['linkID', 'pollutantID', 'emquant_s_cong_2_pct', 'emquant_s_cong_4_pct', 'emquant_s_cong_6_pct',
             'emquant_s_cong_8_pct']], id_vars=['linkID', 'pollutantID'],
    value_vars=['emquant_s_cong_2_pct', 'emquant_s_cong_4_pct', 'emquant_s_cong_6_pct', 'emquant_s_cong_8_pct'])
print(df_cong.groupby(['variable', 'pollutantID'])['value'].mean())
custom_dict = {'CO': 0, 'CO2': 2, 'NOx': 1, 'PM2.5': 3}
df_cong = df_cong.sort_values(by=['linkID', 'pollutantID', 'variable'], key=lambda x: x.map(custom_dict))

sns.set_palette(sns.color_palette('coolwarm', 4))
fig, ax = plt.subplots(figsize=(6.5, 5))
sns.boxplot(df_cong, x='variable', y='value', hue='pollutantID', showfliers=False,
            showmeans=True, meanline=True, meanprops=dict(linestyle='-', linewidth=1, color='purple'), ax=ax)
ax.set_xticklabels([2, 4, 6, 8])
# ax.set_xticks([2, 4, 6, 8])
plt.xlabel('Weeks after announcement')
plt.ylabel('Emission change (%)')
plt.legend(title='')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\E_congestionprice.pdf')

# link-level change
# Total volume change
osm_mt = pd.read_pickle(r'D:\NY_Emission\Shp\osmdta_ritis_signal.pkl')
osm_mt['linkID'] = osm_mt['from_node_'].astype(str) + '_' + osm_mt['to_node_id'].astype(str)
osm_mt = osm_mt.drop_duplicates(subset=['linkID']).reset_index(drop=True)

assign_all = pd.read_pickle(r'D:\NY_Emission\ODME_NY\Simulation_outcome\assign_all_.pkl')
assign_all = assign_all[assign_all['hour'] == 8]
assign_all = assign_all[['from_node_id', 'to_node_id', 'volume_hourly']]
assign_all.columns = ['from_node_', 'to_node_id', 'volume_hourly']
for e_s in ['cg_2', 'cg_4', 'cg_6', 'cg_8']:
    assign_each = pd.read_pickle(r'D:\NY_Emission\ODME_NY\Simulation_outcome\assign_all_%s.pkl' % e_s)
    assign_each = assign_each[assign_each['hour'] == 8]
    assign_each = assign_each[['from_node_id', 'to_node_id', 'volume_hourly']]
    assign_each.columns = ['from_node_', 'to_node_id', 'volume_hourly_%s' % e_s]
    assign_all = assign_all.merge(assign_each, on=['from_node_', 'to_node_id'], how='outer')
    assign_all['volume_pct_%s' % e_s] = 100 * (assign_all['volume_hourly_%s' % e_s] - assign_all['volume_hourly']) / \
                                        assign_all['volume_hourly']

osm_mt = osm_mt.merge(assign_all, on=['from_node_', 'to_node_id'])
osm_mt[['volume_hourly', 'volume_hourly_cg_2', 'volume_hourly_cg_4', 'volume_hourly_cg_6', 'volume_hourly_cg_8']].corr()

geo_congest_price = gpd.read_file(r'D:\NY_Emission\Shp\congest_pricing.shp')
geo_congest_price = geo_congest_price.to_crs(osm_mt.crs)
osm_mt.drop(['index_right'], axis=1, inplace=True)
sinct = gpd.sjoin(osm_mt, geo_congest_price, how="inner", predicate="intersects")
osm_mt['is_cong'] = 0
osm_mt.loc[osm_mt['linkID'].isin(sinct['linkID']), 'is_cong'] = 1

congest_price = gpd.read_file(r'D:\NY_Emission\Shp\congest_pricing.shp')
congest_price = congest_price.to_crs(osm_mt.crs)

osm_mt = osm_mt.replace([np.inf, -np.inf], np.nan)
osm_mt = osm_mt.fillna(0)
osm_mt_amm = osm_mt.copy()
linkids = ['205394_212547', '139_199908', '36991_70', '37461_37462', '36987_37451', '127952_205838', '127909_83',
           '8253_5756', '239542_242107', '204866_204867', '250349_250350', '201655_253985', '201652_201653',
           '260635_260633', '124250_124251', '199490_199502', '199488_224461', '683_684', '205615_37143',
           '237147_237148', '199563_199564']
osm_mt_amm['linkids'] = osm_mt_amm['from_node_'].astype(str) + '_' + osm_mt_amm['to_node_id'].astype(str)
ct_max = [35, 43, 56, 63]
cct = 0
for rr in ['cg_2', 'cg_4', 'cg_6', 'cg_8']:
    binning = mapclassify.NaturalBreaks(-osm_mt_amm['volume_pct_%s' % rr], k=4)  # NaturalBreaks
    osm_mt_amm['cut_jenks'] = (binning.yb + 1) * 0.5
    osm_mt_amm = osm_mt_amm.to_crs('EPSG:32618')
    fig, ax = plt.subplots(figsize=(3.5, 7))
    mpl.rcParams['text.color'] = 'w'  # osm_mt_am['cut_jenks']
    osm_mt_amm.plot(column='volume_pct_%s' % rr, cmap='RdYlGn_r', scheme="user_defined",
                    classification_kwds={'bins': [-15, 0, 15]},
                    lw=1.1, ax=ax, alpha=0.6, legend=True,
                    legend_kwds={'labelcolor': 'white', "fmt": "{:.0f}", 'ncol': 1, 'title': 'CO2 Change (%)',
                                 'loc': 'upper left', 'frameon': True, 'facecolor': 'k', 'edgecolor': 'k',
                                 'framealpha': 0.5})
    mpl.rcParams['text.color'] = 'k'
    congest_price.boundary.plot(ax=ax)
    ctx.add_basemap(ax, crs=osm_mt_amm.crs, source=ctx.providers.CartoDB.DarkMatter, alpha=0.9)
    plt.subplots_adjust(top=0.99, bottom=0.003, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
    plt.axis('off')
    cct += 1
    plt.savefig(r'D:\NY_Emission\Figure\pdiff_VS%s_spatial_change.pdf' % rr)
    plt.close()
