import cv2
from datetime import datetime
from skimage.metrics import structural_similarity
import os
import math
import numpy as np
import pandas as pd
import fastai.vision.all as fva
import pickle
import torch.nn as nn
import torch
from PIL import Image
from torchvision import datasets, transforms
from timm.data.constants import (IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, )
from torch.nn import functional as F
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import timm

# import pathlib
# pathlib.WindowsPath = pathlib.PosixPath

plt.style.use("dark_background")
plt.rcParams.update({"font.size": 22})

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def transform_single(img, input_size=224, model_name='efficientnetv2_rw_t'):
    model_conf = timm.data.resolve_model_data_config(model_name)
    mean = model_conf['mean']
    std = model_conf['std']
    crop_pct = model_conf['crop_pct']
    size = int(input_size / crop_pct)

    data_transforms = transforms.Compose(
        [
            transforms.Resize(
                size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return data_transforms(img)


def count_img(results_ped):
    peds = results_ped[0].boxes.cls
    peds_class = []
    if len(peds) != 0:
        for ii in range(len(peds)):
            label = peds[ii].cpu().numpy().item()
            peds_class.append(results_ped[0].names[int(label)])
    return peds_class


def car_type_predict(cmodel, current_vehicle):
    current_vehicle_pil = Image.fromarray(current_vehicle)
    # cv2.imshow('Frame', current_vehicle)
    x = transform_single(current_vehicle_pil, input_size=224)
    x.unsqueeze_(0)
    output = cmodel.model(x.to(device))
    output_top5 = output.topk(5)[1].cpu().numpy()[0]
    probabilities = F.softmax(output.topk(5).values, dim=1).tolist()[0]
    car_type = cmodel.dls.vocab[output_top5[0]]
    return cmodel.dls.vocab[output_top5[0]], probabilities[0]


def counter_e(vehicle_category, contador_1, WRITE_IMAGE, vehicle_cb, ff, f_c, tdate):
    if vehicle_category in ["car", "Car", "taxi"]:
        contador_1[0] += 1
    if vehicle_category in ['Van', "Pickup"]:
        contador_1[1] += 1
    if vehicle_category in ["Light Truck"]:
        contador_1[2] += 1
    if vehicle_category in ['Heavy Truck']:
        contador_1[3] += 1
    if vehicle_category in ["School Bus"]:
        contador_1[4] += 1
    if vehicle_category in ["bus", "Bus"]:
        contador_1[5] += 1
    if vehicle_category in ["motorcycle"]:
        contador_1[6] += 1
    if vehicle_category in ["truck"]:
        contador_1[7] += 1
    if WRITE_IMAGE:
        cv2.imwrite(r'D:\NY_Emission\Cartype\Crop_test\\' + vehicle_cb + '_' +
                    ff.split('.mp4')[0] + '_' + str(f_c) + '_' + tdate + '_' + str(
            int(datetime.now().timestamp())) + '.jpg', frame[ymin:ymax, xmin:xmax])
    return contador_1


class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0
        self.repeat_count = 0

    def update(self, objects_rect, frame):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x1, y1, x2, y2, conf, classid = rect
            # cx = (x1 + x2) // 2
            # cy = (y1 + y2) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(x1 - pt[0], y1 - pt[1], x2 - pt[2], y2 - pt[3])
                if dist < 35:
                    self.center_points[id] = (x1, y1, x2, y2)
                    #                    print(self.center_points)
                    objects_bbs_ids.append([x1, y1, x2, y2, id, classid])
                    if classid in [2, 5, 6, 7] and (
                            (abs(x1 - x2) * abs(y1 - y2)) / (frame.shape[0] * frame.shape[1])) > 0.001:
                        self.repeat_count += 1
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (x1, y1, x2, y2)
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count, classid])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, _ = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids


# Video para
vdir = r'D:\NY_Emission\Video_Process\NY_Video'
rdir = r'D:\NY_Emission\Video_Process\After2'
cdir = r'D:\NY_Emission\Video_Process\Results2'

allfiles = pd.read_csv(r'D:\NY_Emission\Video_Process\data_models\cams_need.csv')
allfiles = list(allfiles['id'] + '.mp4')
dropfiles = ['39e26a67-8e1d-4760-bd82-59fe3b2db770.mp4', '3f85b7f5-564d-4098-9ffd-580ab6f88ed5.mp4',
             'c9b83e82-d3bf-4235-9cf5-825b400c7322.mp4', '9f6b60c8-b940-4eec-a3c5-4b60d24c751b.mp4',
             '843a0580-a31d-4e5b-90c5-c3875afdfa96.mp4', 'f5c6fd9c-8e5b-4c3c-8c3e-31233678f15b.mp4',
             'f9cb9d4c-10ad-42e4-8997-dbc9e12bd55a.mp4']
allfiles = list(set(allfiles) - set(dropfiles))

print(len(allfiles))
base_dir = r'C:\Users\huson\PycharmProjects\TrafficVedio'
model_dir = r'D:\NY_Emission\Video_Process\data_models'

# %%
WRITE_VIDEO = True
WRITE_IMAGE = False

t_list = pd.date_range(datetime(2023, 12, 5), datetime(2023, 12, 6), freq='h')
t_list = [var.strftime("%Y_%m_%d_%H") for var in t_list]
exist_lt = os.listdir(cdir)

for tdate in t_list[10:11]:
    tdate = '2023_12_05_11'
    # %%
    for ff in tqdm(allfiles[0:1]):
        print(ff)
        # ff = '1572a83a-0a4f-4a7b-84a0-fec0890a2de3.mp4'
        # ff = 'a4c12003-9638-473d-bfe3-dddf509c80b8.mp4'
        # ff = 'GH040143.MP4'
        vidcap = cv2.VideoCapture(f"{vdir}/{tdate}/{ff}")
        I_W = int(vidcap.get(3))
        I_H = int(vidcap.get(4))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        cv2.destroyAllWindows()

        if (I_W > 0) and (f"count_{ff}_{tdate}.csv" not in exist_lt):
            if WRITE_VIDEO:
                out = cv2.VideoWriter(f"{rdir}/{ff[0:-4]}_{tdate}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), int(fps),
                                      (I_W, int(0.9 * I_H)), )

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            CONSIDER_TOP5 = True
            allcount = []
            allloc = []
            with ((torch.no_grad())):
                learn = fva.load_learner(f'{model_dir}/efficientnetv2_rw_t.pkl', cpu=False)
                learn.load_state_dict(torch.load(f'{model_dir}/efficientnetv2_rw_t.pth'))
                learn.to(device)

                # Load yolo models
                model_yolo = YOLO(f'{model_dir}/yolov8x.pt')
                # model_light = YOLO(f'{model_dir}/best_traffic_small_yolo.pt')

                vidcap = cv2.VideoCapture(f"{vdir}/{tdate}/{ff}")
                NUM_FRAMES = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = vidcap.get(cv2.CAP_PROP_FPS)  # fps = 0.5

                # Set traffic counter
                l_ty = 8  # number of car types
                his_track_ids1, his_track_ids2, his_track_ids3, his_track_ids4 = [], [], [], []
                car_types1, car_types2, car_types3, car_types4 = [], [], [], []
                contador_1, contador_2, contador_3, contador_4 = [0] * l_ty, [0] * l_ty, [0] * l_ty, [0] * l_ty
                contador_1_a, contador_2_a, contador_3_a, contador_4_a = 0, 0, 0, 0
                lin1, lin2, lin3, lin4 = int(I_H * 0.8), int(I_H * 0.7), int(I_H * 0.6), int(I_H * 0.5)

                # Load tracker
                f_c = 0
                ret = True
                tracker1 = Tracker()

                while ret:
                    print(f"{f_c}/{NUM_FRAMES}", end="\r")
                    ret, frame = vidcap.read()
                    if f_c >= NUM_FRAMES: break
                    if not ret and frame is None:
                        ret = True
                        print('~~~~~~~~~~~~~~~Frame None Type~~~~~~~~~~~~~~~~~~~~~~')
                        continue
                    # if not ret: break
                    # Detect items based on YOLO
                    frame = frame[int(I_H * 0.1):int(I_H * 1), 0:frame.shape[1]]
                    results_ped = model_yolo(frame, conf=0.3, device=device)
                    # for r in results_ped: frame_pt = r.plot()
                    # cv2.imshow('frame', frame_pt)

                    predictions = results_ped[0]
                    # pd.DataFrame([predictions.names]).T.to_csv('temp.csv')
                    boxes = predictions.boxes.xyxy  # x1, y1, x2, y2
                    scores = predictions.boxes.conf
                    categories = predictions.boxes.cls

                    # Count all detected classes
                    peds_class = count_img(results_ped)
                    # peds_class = {}

                    # Analyze detected objects
                    dets = []
                    all_models = []
                    all_lights = []
                    frame_bk = frame.copy()
                    num_queue = 0
                    if len(boxes) != 0:
                        # Extract all detected objects related to traffic
                        for i in range(len(boxes)):
                            xmin, ymin, xmax, ymax = boxes[i].cpu().numpy().astype(int)
                            conf = scores[i].cpu().numpy()
                            label = categories[i].cpu().numpy().astype(int).item()
                            dets.append([xmin, ymin, xmax, ymax, float(conf), label])
                            cv2.rectangle(frame_bk, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

                        # Tracking
                        dets = np.asarray(dets)
                        try:
                            tracks = tracker1.update(dets, frame)
                            # tracks = dets.copy()
                        except Exception as ee:
                            print(ee)

                        # Extract all tracked objects
                        num_queue = tracker1.repeat_count  # queuing length
                        boxes = []
                        indexIDs = []
                        classIDs = []
                        for track in tracks:
                            boxes.append([track[0], track[1], track[2], track[3]])
                            indexIDs.append(int(track[4]))
                            classIDs.append(predictions.names[int(track[5])])

                        if len(boxes) > 0:
                            i = int(0)
                            for box in boxes:
                                # extract the bounding box coordinates
                                (xmin, ymin) = (int(box[0]), int(box[1]))
                                (xmax, ymax) = (int(box[2]), int(box[3]))
                                width_box = xmax - xmin
                                height_box = ymax - ymin
                                # print((width_box * height_box) / (frame.shape[0] * frame.shape[1]))
                                center_box = (int(width_box / 2), int(height_box / 2))
                                tracked_id = indexIDs[i]
                                vehicle_category = classIDs[i]

                                if vehicle_category in ['car', 'bus', 'truck', "motorcycle"]:

                                    # Change the vehicle_category based on our own model: only when touch the line
                                    if (((ymin < lin1) and (ymax > lin1)) | ((ymin < lin2) and (ymax > lin2)) | (
                                            (ymin < lin3) and (ymax > lin3)) | ((ymin < lin4) and (ymax > lin4))):
                                        car_t, car_p = car_type_predict(cmodel=learn,
                                                                        current_vehicle=frame[ymin:ymax, xmin:xmax])
                                        if vehicle_category == 'motorcycle':
                                            car_tn = vehicle_category
                                        else:
                                            car_tn = car_t
                                    else:
                                        car_tn = vehicle_category

                                    vehicle_cb = vehicle_category + '_' + car_tn

                                    # Checking if the center of recognized vehicle touch the detector line
                                    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    if (ymin < lin1) and (ymax > lin1) and (tracked_id not in his_track_ids1):
                                        contador_1 = counter_e(car_tn, contador_1, WRITE_IMAGE, vehicle_cb, ff, f_c,
                                                               tdate)
                                        contador_1_a = sum(contador_1)
                                        his_track_ids1.append(tracked_id)
                                    if (ymin < lin2) and (ymax > lin2) and (tracked_id not in his_track_ids2):
                                        contador_2 = counter_e(car_tn, contador_2, WRITE_IMAGE, vehicle_cb, ff, f_c,
                                                               tdate)
                                        contador_2_a = sum(contador_2)
                                        his_track_ids2.append(tracked_id)
                                    if (ymin < lin3) and (ymax > lin3) and (tracked_id not in his_track_ids3):
                                        contador_3 = counter_e(car_tn, contador_3, WRITE_IMAGE, vehicle_cb, ff, f_c,
                                                               tdate)
                                        contador_3_a = sum(contador_3)
                                        his_track_ids3.append(tracked_id)
                                    if (ymin < lin4) and (ymax > lin4) and (tracked_id not in his_track_ids4):
                                        contador_4 = counter_e(car_tn, contador_4, WRITE_IMAGE, vehicle_cb, ff, f_c,
                                                               tdate)
                                        contador_4_a = sum(contador_4)
                                        his_track_ids4.append(tracked_id)
                                    if WRITE_VIDEO:
                                        if vehicle_category in ['car', 'bus', 'truck']:
                                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 1)
                                        elif vehicle_category in ['person', 'bicycle', 'motorcycle']:
                                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                                i += 1
                            # cv2.imshow('Frame', frame)

                            if WRITE_VIDEO:
                                cv2.line(frame, (0, lin1), (I_W, lin1), (255, 255, 100), 1)
                                cv2.line(frame, (0, lin2), (I_W, lin2), (255, 255, 100), 1)
                                cv2.line(frame, (0, lin3), (I_W, lin3), (255, 255, 100), 1)
                                cv2.line(frame, (0, lin4), (I_W, lin4), (255, 255, 100), 1)
                                # fontFace = cv2.FONT_HERSHEY_TRIPLEX,
                                cv2.putText(img=frame, text=f'{contador_1_a}', org=(int(I_H * 0.1), lin1 + 2),
                                            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                            fontScale=0.6, color=(0, 255, 255), thickness=1)
                                cv2.putText(img=frame, text=f'{contador_2_a}', org=(int(I_H * 0.1), lin2 + 2),
                                            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                            fontScale=0.6, color=(0, 255, 255), thickness=1)
                                cv2.putText(img=frame, text=f'{contador_3_a}', org=(int(I_H * 0.1), lin3 + 2),
                                            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                            fontScale=0.6, color=(0, 255, 255), thickness=1)
                                cv2.putText(img=frame, text=f'{contador_4_a}', org=(int(I_H * 0.1), lin4 + 2),
                                            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                            fontScale=0.6, color=(0, 255, 255), thickness=1)

                        # cv2.imshow('Frame', frame)

                    # Compute similarity between frames
                    frame_bk = cv2.cvtColor(frame_bk, cv2.COLOR_BGR2GRAY)
                    score = np.nan
                    if f_c > 0: (score, diff) = structural_similarity(frame_bk, last_bk, full=True)

                    torch.cuda.empty_cache()
                    if WRITE_VIDEO:
                        cv2.putText(img=frame, text=f'{round(score, 3)}', org=(int(I_H * 0.1), int(I_H * 0.1)),
                                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                    fontScale=0.6, color=(0, 255, 255), thickness=1)
                        out.write(frame)

                    allcount.append(
                        [ff, f_c, contador_1.copy(), contador_2.copy(), contador_3.copy(), contador_4.copy(),
                         dict(Counter(peds_class)), score, num_queue, tdate])
                    f_c += 1
                    last_bk = frame_bk

            allcount = pd.DataFrame(allcount)
            allcount.columns = ['file', 'frame', 'c1', 'c2', 'c3', 'c4', 'peds', 'similar',
                                'num_repeat', 'Date_hour']

            # allcount.to_csv(f"{cdir}/count_{ff}_{tdate}.csv")

            # %%
            if WRITE_VIDEO:
                out.release()
