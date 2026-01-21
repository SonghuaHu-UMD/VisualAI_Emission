#!/usr/bin/python;
import os, glob
import math
from math import atan, sin
import sys
import pandas as pd
import csv
import time
from tqdm import tqdm
import gc
from pathlib import Path

start_time = time.time()

## this path may need to be changed to locate matrix database directory
# /data/input_s_cong_8
matrixdatapath = r"/data/MOVES"
# hf: month from 8 to 1
# ess = ['s_raw', 's_peak10', 's_remote10', 's_mode10']  # 'cd', 'te', 'hf', 'ss', '', 'ns', 'nsp', 'nvo'
# ess = ['s_peak20', 's_remote20', 's_mode20']
# ess = ['s_peak30', 's_remote30', 's_mode30']
# ess=['ncamera']
ess=['s_cong_8']

# <codecell>
def main():
    calctime = 0
    print("Let's hang out with MOVES-Matrix! " + "Updated on November 30, 2024.")
    print()
    for kk in ess:
        print(kk)
        path = r"/data/input_%s" % kk
        os.chdir(path)
        with open('batchmode.csv', 'rt') as f:
            readbatch = csv.reader(f, delimiter=',')
            next(readbatch)
            for batchrow in readbatch:
                print("")
                print("Starting task" + batchrow[0])
                matrixdataname = locatematrixdata(batchrow[0], batchrow[1], batchrow[2], batchrow[3].split(".csv")[0],
                                                  path)
                importmatrixdb(batchrow[1], matrixdataname)
                tt1 = time.time()
                importinputcsv(batchrow[2].split(".csv")[0], batchrow[4].split(".csv")[0], batchrow[5].split(".csv")[0],
                               batchrow[6].split(".csv")[0], batchrow[7].split(".csv")[0], batchrow[8].split(".csv")[0],
                               path)
                if batchrow[8].lower() in ('d'):
                    print("second-by-second schedule is imported for vehicle operation...")
                    calcopmode(batchrow[7].split(".csv")[0], path, batchrow[0])
                elif batchrow[8].lower() in ('o'):
                    print("opmode distribution is imported for vehicle operation...")
                    importopmode(batchrow[7].split(".csv")[0], path)
                else:
                    print("emission is generated based on MOVES default schedules...")
                    avgspeedemissioncalc(path)
                emissioncalc()
                tt2 = time.time()
                print("Exporting result for task" + batchrow[0] + "..." + path)
                resultoutput(batchrow[0], path)
                calctime += round((tt2 - tt1), 2)
        print("")
        print(
            "All tasks in batch mode list have been finished. You can find your results in the output folder under your working directory.")
        print("--- Calculation Finished in %s seconds ---" % calctime)
        print("--- Total Finished in %s seconds ---" % round((time.time() - start_time), 2))


# <codecell>
def avgspeedemissioncalc(path):
    global mx, sourcetypeagedistributiondf, opmodedistributiondf, linksourcetypehourdf, linkdf, emissioninventory, eminvbylinksource
    os.chdir(path)
    global default_opmode_table
    default_opmode_table = pd.read_csv('default_opmode_project.csv', header=0)
    linkdf.loc[linkdf.linkAvgSpeed > 80, 'linkAvgSpeed'] = 80.0
    linkdf.loc[linkdf.linkAvgSpeed < 1, 'linkAvgSpeed'] = 1.0
    linkdf['linkAvgSpeed'] = linkdf[['linkAvgSpeed']].apply(lambda x: pd.Series.round(x, 1))
    opmodedistributiondf = pd.merge(default_opmode_table, linkdf, how='inner', on=['roadTypeID', 'linkAvgSpeed'])
    opmodedistributiondf = opmodedistributiondf[['sourceTypeID', 'linkID', 'opModeID', 'opModeFraction']]


# <codecell>

def locatematrixdata(idid, region, year, meteo, path):
    os.chdir(path)
    with open(meteo + '.csv', 'rt') as meteofile:
        rd = csv.reader(meteofile)  # comma is default delimiter
        next(rd)
        tempdata = next(rd)
        month = int(tempdata[0])
        wintermonth = [11, 12, 1, 2, 3]
        transmonth = [4, 10]
        if month in wintermonth:
            month = 1
        elif month in transmonth:
            month = 4
        else:
            month = 7
        T = float(tempdata[3])
        H = float(tempdata[4])
        if T < 10 or T > 110 or T % 5 != 0.0:
            print(
                "In task " + idid + ", Temperature needs to be set between 5F and 110F, and with interval of 5F, like: 10, 15, 20...")
            print("The program is ended, please reset the meteorology input and rerun.")
            sys.exit(0)
        elif H < 0 or H > 100 or H % 5 != 0.0:
            print(
                "In task " + idid + ", Humidity needs to be set between 0% and 100%, and with interval of 5%, like: 0, 5, 10, 15, 20...")
            print("The program is ended, please reset the meteorology input and rerun.")
            sys.exit(0)
        # T = min(max(int(T0/5)*5+int((T0%5)/2.5)*5,10),110)
        # H = min(max(int(H0/5)*5+int((H0%5)/2.5)*5,0),100)
        matrixname = "" + str(year) + "_" + str(month) + "_" + str(int(T)) + "_" + str(int(H))
        print("The " + region + " database " + matrixname + " is used. " + "The input temperature is " + str(
            T) + "F, and input Humidity is set as " + str(H) + "%.")
        return matrixname


# <codecell>

def importmatrixdb(region, matrixdataname):
    global con
    global cur
    os.chdir(matrixdatapath + '/' + region.lower())
    for fn in glob.glob(matrixdatapath + '/' + region.lower() + '/*_' + matrixdataname + '.csv'):
        dr = pd.read_csv(fn, header=None)
    dr.columns = ['opModeID', 'pollutantID', 'sourceTypeID', 'modelYearID', 'em', 'hehe']
    pollist = [1, 2, 3, 87, 90, 91, 98, 100, 110, '1', '2', '3', '87', '90', '91', '98', '100', '110']
    global mx
    mx = dr[(dr['pollutantID'].isin(pollist))].drop(['hehe'], axis=1)
    # print mx


# <codecell>

def importinputcsv(year, linksourcetypehour, sourcetypeagedistribution, link, opcycle, method, path):
    os.chdir(path)
    global linkdf, sourcetypeagedistributiondf, linksourcetypehourdf, opcycledf, linkspeeddf
    linkdf = pd.read_csv(link + ".csv", header=0)
    sourcetypeagedistributiondf = pd.read_csv(sourcetypeagedistribution + ".csv", header=0)
    # print sourcetypeagedistributiondf
    sourcetypeagedistributiondf['yearID'] = int(year)
    sourcetypeagedistributiondf['modelYearID'] = sourcetypeagedistributiondf.yearID - sourcetypeagedistributiondf.ageID
    # print sourcetypeagedistributiondf
    linksourcetypehourdf = pd.read_csv(linksourcetypehour + ".csv", header=0)
    if method == 'd':
        opcycledf = pd.read_csv(opcycle + ".csv", header=0)
        linkspeeddf = opcycledf.groupby(['linkID']).mean()
        linkspeeddf = pd.DataFrame(linkspeeddf.drop(['secondID', 'grade'], axis=1).reset_index())


# <codecell>

def importopmode(opmodedistribution, path):
    os.chdir(path)
    global opmodedistributiondf
    opmodedistributiondf = pd.read_csv(opmodedistribution + ".csv", header=0)

#    opmodedistributiondf.to_csv("opInput_task.csv")


# <codecell>
def calcopmode(driveschedulesecondlink, path, taskid):
    sourcetype = [11, 21, 31, 32, 41, 42, 43, 51, 52, 53, 54, 61, 62]
    sourcetypeparam = []
    sourcetypeparam.append([0.0251, 0, 0.000315, 0.285, 0.285])  # 11
    sourcetypeparam.append([0.156461, 0.00200193, 0.000492646, 1.4788, 1.4788])
    sourcetypeparam.append([0.22112, 0.00283757, 0.000698282, 1.86686, 1.86686])
    sourcetypeparam.append([0.235008, 0.00303859, 0.000747753, 2.05979, 2.05979])
    sourcetypeparam.append([1.29515, 0, 0.00371491, 19.5937, 17.1])
    sourcetypeparam.append([1.0944, 0, 0.00358702, 16.556, 17.1])  ###42
    sourcetypeparam.append([0.746718, 0, 0.00217584, 9.06989, 17.1])
    sourcetypeparam.append([1.58346, 0, 0.00357228, 23.1135, 17.1])
    sourcetypeparam.append([0.627922, 0, 0.00160302, 8.53896, 17.1])
    sourcetypeparam.append([0.557262, 0, 0.00147383, 6.98448, 17.1])
    sourcetypeparam.append([0.68987, 0, 0.00210545, 7.52572, 17.1])
    sourcetypeparam.append([1.53819, 0, 0.00403054, 22.9745, 17.1])
    sourcetypeparam.append([1.63041, 0, 0.00418844, 24.601, 17.1])
    os.chdir(path)

    drivecycle = pd.read_csv(driveschedulesecondlink + ".csv").head(10000)
    drivecycle['acc'] = drivecycle.groupby('linkID')['speed'].diff().fillna(0)
    sch_linkid = drivecycle['linkID'].tolist()
    # sch_id = drivecycle['secondID'].tolist()
    sch_speed = drivecycle['speed'].tolist()
    sch_grade = drivecycle['grade'].tolist()
    sch_acc = drivecycle['acc'].tolist()
    del drivecycle

    print('Starting VSP and opmode calculation ...')
    gc.collect()
    global opmodedistributiondf
    opmodedistributiondf = pd.DataFrame()
    for jj1 in [12]: # tqdm(range(0, 13))
        temp_opmode = {}
        temp_vsp = [0 for x in range(len(sch_speed))]
        for jj2 in range(0, len(sch_speed)):
            key = (sourcetype[jj1], sch_linkid[jj2], jj2)
            temp_vsp[jj2] = vspcalc(round(sourcetypeparam[jj1][0], 12), round(sourcetypeparam[jj1][1], 12), \
                                    round(sourcetypeparam[jj1][2], 12), round(sourcetypeparam[jj1][3], 12), \
                                    round(sourcetypeparam[jj1][4], 12), sch_speed[jj2], sch_acc[jj2],
                                    sch_grade[jj2])
            if sch_speed[jj2] < 1:
                temp_opmode[key] = int(1)
                # temp_opmode[jj1][jj2]=int(1)
            elif sch_acc[jj2] + (9.81 / 0.44704 * sin(atan(sch_grade[jj2] / 100.0))) <= -2.0:
                temp_opmode[key] = int(0)
                # temp_opmode[jj1][jj2]=int(0)
            elif jj2 >= 2 and sch_linkid[jj2] == sch_linkid[jj2 - 1] and sch_linkid[jj2] == sch_linkid[jj2 - 2] \
                    and sch_acc[jj2] + (9.81 / 0.44704 * sin(atan(sch_grade[jj2] / 100.0))) < -1 \
                    and sch_acc[jj2 - 1] + (9.81 / 0.44704 * sin(atan(sch_grade[jj2 - 1] / 100.0))) < -1 \
                    and sch_acc[jj2 - 2] + (9.81 / 0.44704 * sin(atan(sch_grade[jj2 - 2] / 100.0))) < -1:
                temp_opmode[key] = int(0)
                # temp_opmode[jj1][jj2]=int(0)
            elif sch_speed[jj2] < 25:
                if temp_vsp[jj2] < 0:
                    temp_opmode[key] = int(11)  # temp_opmode[jj1][jj2]=int(11)
                elif temp_vsp[jj2] < 3:
                    temp_opmode[key] = int(12)  # temp_opmode[jj1][jj2]=int(12)
                elif temp_vsp[jj2] < 6:
                    key = (sourcetype[jj1], sch_linkid[jj2], jj2)
                    temp_opmode[key] = int(13)  # temp_opmode[jj1][jj2]=int(13)
                elif temp_vsp[jj2] < 9:
                    temp_opmode[key] = int(14)  # temp_opmode[jj1][jj2]=int(14)
                elif temp_vsp[jj2] < 12:
                    temp_opmode[key] = int(15)  # temp_opmode[jj1][jj2]=int(15)
                else:
                    temp_opmode[key] = int(16)  # temp_opmode[jj1][jj2]=int(16)
            elif sch_speed[jj2] < 50:
                if temp_vsp[jj2] < 0:
                    temp_opmode[key] = int(21)  # temp_opmode[jj1][jj2]=int(21)
                elif temp_vsp[jj2] < 3:
                    temp_opmode[key] = int(22)  # temp_opmode[jj1][jj2]=int(22)
                elif temp_vsp[jj2] < 6:
                    temp_opmode[key] = int(23)  # temp_opmode[jj1][jj2]=int(23)
                elif temp_vsp[jj2] < 9:
                    temp_opmode[key] = int(24)  # temp_opmode[jj1][jj2]=int(24)
                elif temp_vsp[jj2] < 12:
                    temp_opmode[key] = int(25)  # temp_opmode[jj1][jj2]=int(25)
                elif temp_vsp[jj2] < 18:
                    temp_opmode[key] = int(27)  # temp_opmode[jj1][jj2]=int(27)
                elif temp_vsp[jj2] < 24:
                    temp_opmode[key] = int(28)  # temp_opmode[jj1][jj2]=int(28)
                elif temp_vsp[jj2] < 30:
                    temp_opmode[key] = int(29)  # temp_opmode[jj1][jj2]=int(29)
                else:
                    temp_opmode[key] = int(30)  # temp_opmode[jj1][jj2]=int(30)
            else:
                if temp_vsp[jj2] < 6:
                    temp_opmode[key] = int(33)  # temp_opmode[jj1][jj2]=int(33)
                elif temp_vsp[jj2] < 12:
                    temp_opmode[key] = int(35)  # temp_opmode[jj1][jj2]=int(35)
                elif temp_vsp[jj2] < 18:
                    temp_opmode[key] = int(37)  # temp_opmode[jj1][jj2]=int(37)
                elif temp_vsp[jj2] < 24:
                    temp_opmode[key] = int(38)  # temp_opmode[jj1][jj2]=int(38)
                elif temp_vsp[jj2] < 30:
                    temp_opmode[key] = int(39)  # temp_opmode[jj1][jj2]=int(39)
                else:
                    temp_opmode[key] = int(40)  # temp_opmode[jj1][jj2]=int(40)

        temp_opmodedf = pd.Series(temp_opmode).reset_index()
        temp_opmodedf.columns = ['sourceTypeID', 'linkID', 'SecondID', 'opModeID']
        temp_opmodedf = temp_opmodedf.drop('SecondID', axis=1)

        opmode_cte = temp_opmodedf.groupby(['sourceTypeID', 'linkID', 'opModeID']).size().reset_index(name='opmodect')
        opmode_cte['opModeFraction'] = opmode_cte['opmodect'] / opmode_cte.groupby(['sourceTypeID', 'linkID'])[
            'opmodect'].transform('sum')
        opmodedistributiondf = pd.concat([opmodedistributiondf, opmode_cte], axis=0)
    # cur.execute("CREATE TABLE opmode_num (sourcetypeID int, linkID int, opModeID int);")

    print('Finish VSP and opmode calculation !!')
    del temp_vsp, temp_opmodedf
    gc.collect()

    opmodedistributiondf.to_csv("median_opemode_%s.csv" % taskid)
    opmodedistributiondf = opmodedistributiondf.round({'opModeFraction': 5})


# <codecell>
def vspcalc(a, b, c, m, M, speed, acc, grade):
    return round((a / M * speed * 0.44704 + b / M * (speed * 0.44704) ** 2 + c / M * (speed * 0.44704) ** 3 + (
            m / M) * speed * 0.44704 * (9.81 * math.sin(math.atan(0.01 * grade)) + acc * 0.44704)), 12)


# <codecell>
def emissioncalc():
    global mx, sourcetypeagedistributiondf, opmodedistributiondf, linksourcetypehourdf, linkdf
    embysource_2_opmode = pd.merge(mx, sourcetypeagedistributiondf, how='inner', on=['modelYearID', 'sourceTypeID'])
    embysource_2_opmode['ageem'] = embysource_2_opmode.ageFraction * embysource_2_opmode.em
    embysource_2_opmode = embysource_2_opmode.groupby(['sourceTypeID', 'pollutantID', 'opModeID']).agg(
        {'ageem': 'sum'}).reset_index()
    embylinksourceop = pd.merge(opmodedistributiondf, embysource_2_opmode, how='inner', on=['sourceTypeID', 'opModeID'])
    embylinksourceop['embyopmodefra'] = embylinksourceop.opModeFraction * embylinksourceop.ageem
    emhrbylinksource = embylinksourceop.groupby(['linkID', 'sourceTypeID', 'pollutantID']).agg(
        {'embyopmodefra': 'sum'}).reset_index()
    # emhrbylinksource.to_csv(path + 'embysource.csv',index = False)
    # linksourcetypehourdf.to_csv(path + 'linksourcetypehourdf.csv',index = False)
    embylinksource_temp = pd.merge(emhrbylinksource, linksourcetypehourdf, how='inner', on=['sourceTypeID', 'linkID'])
    # embylinksource_temp.to_csv(path + 'embylinksource_temp.csv',index = False)
    embylinksource_temp['emissions'] = embylinksource_temp.embyopmodefra * embylinksource_temp.sourceTypeHourFraction
    emhourbylink = embylinksource_temp.groupby(['linkID', 'pollutantID']).agg({'emissions': 'sum'}).reset_index()
    emhourbylinksource = embylinksource_temp.groupby(['linkID', 'pollutantID', 'sourceTypeID']).agg(
        {'emissions': 'sum'}).reset_index()

    global emissioninventory, eminvbylinksource
    emissioninventory = pd.merge(emhourbylink, linkdf, how='inner', on=['linkID'])

    emissioninventory[
        'emquant'] = emissioninventory.emissions * emissioninventory.linkVolume * emissioninventory.linkLength / emissioninventory.linkAvgSpeed
    emissioninventory[
        'emrate'] = emissioninventory.emissions * emissioninventory.linkVolume / emissioninventory.linkAvgSpeed
    emissioninventory = emissioninventory.groupby(['linkID', 'pollutantID']).agg(
        {'emquant': 'sum', 'emrate': 'sum'}).reset_index()
    eminvbylinksource = pd.merge(emhourbylinksource, linkdf, how='inner', on=['linkID'])
    # eminvbylinksource.to_csv(path + 'eminvbylinksource.csv',index = False)
    eminvbylinksource[
        'emquant'] = eminvbylinksource.emissions * eminvbylinksource.linkVolume * eminvbylinksource.linkLength / eminvbylinksource.linkAvgSpeed
    eminvbylinksource[
        'emrate'] = eminvbylinksource.emissions * eminvbylinksource.linkVolume / eminvbylinksource.linkAvgSpeed
    eminvbylinksource.dropna(subset=['emrate'], inplace=True)
    eminvbylinksource.reset_index()
    eminvbylinksource = eminvbylinksource.groupby(['linkID', 'sourceTypeID', 'pollutantID']).agg(
        {'emquant': 'sum', 'emrate': 'sum'}).reset_index()


# <codecell>
def resultoutput(idid, path):
    checkpoint_dir = Path(path + '/Output')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(path + '/Output')
    emissioninventory1 = emissioninventory[['linkID', 'pollutantID', 'emrate', 'emquant']]
    eminvbylinksource1 = eminvbylinksource[['linkID', 'sourceTypeID', 'pollutantID', 'emrate', 'emquant']]
    print(emissioninventory1)
    emissioninventory1.to_csv('task' + str(idid) + '_emissionbylink.csv', sep=',', index=False)
    eminvbylinksource1.to_csv('task' + str(idid) + '_emissionbylinksource.csv', sep=',', index=False)


# <codecell>


main()
