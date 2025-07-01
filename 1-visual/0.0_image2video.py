# %% Image crawler from NYDOT
import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor
import json
from os import listdir
from os.path import isfile, join
import pickle

# Get all locations
cams = pd.DataFrame(requests.get(r'https://webcams.nyctmc.org/api/cameras/').json())
cams = cams.sort_values(by='id', ascending=True).reset_index(drop=True)
url_list = list(cams['imageUrl'])
print(len(url_list))
cams['area'].value_counts()

# Remove poor video
pdir = r'D:\NY_Emission\Poor'
poorfiles = [f[0:-4] for f in listdir(pdir) if isfile(join(pdir, f))]
cams = cams[~cams['id'].isin(poorfiles)]

# 5s video
s5dir = r'D:\NY_Emission\Video_Process\Gap_5seconds'
s5files = [f[0:-8] for f in listdir(s5dir) if isfile(join(s5dir, f))]
cams['tt'] = 2
cams.loc[cams['id'].isin(s5files), 'tt'] = 5

# matched video
with open(r'D:\NY_Emission\matched.pickle', 'rb') as handle:
    matched = pickle.load(handle)

# Need: Manhattan + Matched
cams_need = cams[(cams['area'] == 'Manhattan') | (cams['id'].isin(matched))].reset_index(drop=True)
cams_need.to_csv(r'D:\NY_Emission\cams_need.csv')

# Get camera that need to crawl
with open(r'D:\NY_Emission\NY_Emission-0910.json') as f: traffic_count = json.load(f)
traffic_count = pd.DataFrame(traffic_count['dataset']['samples'])
traffic_count['name'] = traffic_count['name'].str.replace('.jpg', '')


# Only those with matched
def get_url(url):
    while True:
        try:
            img = requests.get(url).content
            with open(r'E:\Traffic_camera\NY\%s_%s.jpg' % (url.split('/')[-2], time.time()), 'wb') as h:
                h.write(img)
            time.sleep(2)
        except:
            time.sleep(2)


with ThreadPoolExecutor(max_workers=len(url_list)) as exector: exector.map(get_url, url_list)

# %% Image2video
import pandas as pd
import requests
import moviepy.video.io.ImageSequenceClip
import os
import glob

print(os.getcwd())
import multiprocessing

print(multiprocessing.cpu_count())
import matplotlib as mpl

print(mpl.get_cachedir())

# Get all locations
cams = pd.DataFrame(requests.get(r'https://webcams.nyctmc.org/api/cameras/').json())
url_list = list(cams['imageUrl'])
print(len(url_list))

# %system top -b -n 1 | grep Cpu
# %system free -m

# get all image files
i_f = r'/home/hu/NY_Image//'
files_raw = pd.DataFrame(glob.glob(r'%s*' % i_f))
# files_raw.to_csv(r'filename.csv')

files = files_raw[0].str.split('/', expand=True)
files['URL'] = files_raw[0]
files_names = files[files.columns[-1]].str.split('_', expand=True)
files['Timestamp'] = files_names[files_names.columns[-1]].str.split('.jpg', expand=True)[0].astype(float)
files['Datetime'] = pd.to_datetime(files['Timestamp'], unit='s', utc=True).dt.tz_convert('US/Eastern').dt.tz_localize(
    None).dt.ceil(freq='s')
files['ID'] = files_names[1].str.split('/', expand=True)[1]
files = files.sort_values(by=['ID', 'Datetime']).reset_index(drop=True)
files = files[['URL', 'ID', 'Timestamp', 'Datetime']]
files['Diff'] = files.groupby('ID')['Timestamp'].diff()
# files['Diff'].describe()
camera_count = files.groupby('ID')['Timestamp'].count().sort_values().reset_index()
print('No of camera: %s, Mean interval: %s' % (len(camera_count), files['Diff'].mean()))

# # Whether time increases over time
# files['Date']=files['Datetime'].dt.date
# files['Hour']=files['Datetime'].dt.hour
# files.groupby(['Date','Hour'])['Diff'].mean()
# files['Datetime'].max()
camera_count['Timestamp'].plot(marker='o', markersize=2, color='k')

# Build vedio and select
# i_f = r'/home/hu/NY_Image//'
# files = pd.DataFrame(glob.glob(r'%s*' % i_f))
files = pd.read_csv('files.csv')
all_ids = list(set(files['ID']))
error_list = []
for kk in all_ids:
    image_files = list(files.loc[files['ID'] == kk, 'URL'])
    image_files.sort()
    try:
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files[0:600], fps=1)
        clip.write_videofile('/home/hu/NY_Vedio//%s.mp4' % kk)
    except:
        print(kk)
        error_list.append(kk)
