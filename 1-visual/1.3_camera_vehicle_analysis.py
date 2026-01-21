import torch
import pandas as pd
import glob
import cv2
import timm
import os
import fastai.vision.all as fva
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from adjustText import adjust_text
import seaborn as sns
import numpy as np
import disarray

pd.options.mode.chained_assignment = None

def transform_single(img, input_size=224, crop_pct=1, mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
                     std=timm.data.constants.IMAGENET_INCEPTION_STD):
    if crop_pct is None:
        crop_pct = 224 / 256
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


# Plot model performance
file_name = glob.glob(r'D:\NY_Emission\Cartype\f_model\plot_focal\*')
all_train_loss = pd.DataFrame()
all_val_loss = pd.DataFrame()
for kk in file_name:
    model_name = kk.split('\\')[-1].split('.')[0].split('-')[1]
    loss = pd.read_pickle(kk + '\\metrics_train.pkl')
    val_loss = pd.read_pickle(kk + '\\metrics_val.pkl')
    val_loss['model_name'] = model_name
    loss['model_name'] = model_name
    # loss.columns = [model_name, 'para', 'time']
    # loss = loss[[model_name]]
    all_train_loss = pd.concat([all_train_loss, loss], axis=0)
    all_val_loss = pd.concat([all_val_loss, val_loss], axis=0)

# Plot the model: Training
all_train_loss = all_train_loss[all_train_loss['model_name'].isin(
    ['vit_small_patch16_224', 'swin_base_patch4_window7_224', 'convnext_tiny', 'repvgg_a2', 'inception_v4', 'resnet50',
     'densenet201', 'inception_next_tiny', 'xception71', 'efficientnetv2_rw_t'])]
all_train_loss['model_name'] = all_train_loss['model_name'].map(
    {'vit_small_patch16_224': 'ViT', 'swin_base_patch4_window7_224': 'Swin', 'convnext_tiny': 'ConvNeXt',
     'repvgg_a2': 'RepVGG', 'inception_v4': 'Inception-v4', 'resnet50': 'ResNet-50', 'densenet201': 'DenseNet-201',
     'inception_next_tiny': 'InceptionNeXt', 'efficientnet_b3': 'EfficientNet', 'xception71': 'Xception',
     'efficientnetv2_rw_t': 'EfficientNet-v2'})
all_train_loss['time'] = all_train_loss['time'].dt.total_seconds()
all_train_loss = all_train_loss.reset_index()
all_train_loss['time'] = all_train_loss['time'] / (all_train_loss['index'].max() + 1)
all_train_loss['time'] = all_train_loss['time'] * all_train_loss['index']
all_train_loss['time'] = all_train_loss['time'].astype(int)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
all_train_lossp = pd.pivot_table(all_train_loss.reset_index(), values='loss', index=['time'], columns=['model_name'])
# all_val_loss.groupby(['model_name'])['train_loss'].plot(legend=True, ax=ax)
all_train_lossp.fillna(method='bfill').plot(colormap='tab20', ax=ax, lw=2)
plt.legend(loc='upper right', ncol=2, frameon=False)
plt.xlim([0, 3000])
plt.ylabel('Training loss')  # (Cross Entropy)
plt.xlabel('Training time (s)')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\train_loss.pdf')

# Plot by Parameters size
model_acc = all_val_loss.groupby('model_name')['accuracy'].max().sort_values().reset_index()
model_acc['model_name'] = model_acc['model_name'].map(
    {'vit_small_patch16_224': 'ViT', 'swin_base_patch4_window7_224': 'Swin', 'convnext_tiny': 'ConvNeXt',
     'repvgg_a2': 'RepVGG', 'inception_v4': 'Inception-v4', 'resnet50': 'ResNet-50', 'densenet201': 'DenseNet-201',
     'inception_next_tiny': 'InceptionNeXt', 'efficientnet_b3': 'EfficientNet', 'xception71': 'Xception',
     'efficientnetv2_rw_t': 'EfficientNet-v2'})
model_acc = model_acc.merge(all_train_loss.drop_duplicates('model_name'), on='model_name')
model_acc['accuracy'] = model_acc['accuracy'] * 100
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(model_acc['para'], model_acc['accuracy'], 'o', markersize=10, color='blue', alpha=0.5)
ts = []
for i in range(10):
    ts.append(plt.text(model_acc.loc[i, 'para'], model_acc.loc[i, 'accuracy'], model_acc.loc[i, 'model_name']))
# adjust_text(ts, x=model_acc['para'], y=model_acc['accuracy'], force_points=0.2,
#             arrowprops=dict(arrowstyle='->', color='green'))
plt.ylabel('Top-1 Accuracy (%)')
plt.xlabel('Number of Parameters (Millions)')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\parasize_loss.pdf')

# Final model
model_nm = 'convnext_tiny'
# Plot confusion_matrix
confusion_matrix = pd.read_pickle(r'D:\NY_Emission\Cartype\f_model\20240508210959-%s\confusion_matrix.pkl' % model_nm)
confusion_matrix = confusion_matrix.astype(int)
# ndd = ['Passenger_Car', 'Passenger_Truck', 'Intercity_Bus', 'Transit_Bus', 'Motorcycle', 'Combination_Truck', 'EV']
redd = {'Passenger_Car': 'P_Car', 'Passenger_Truck': 'P_Truck', 'Intercity_Bus': 'I_Bus',
        'Transit_Bus': 'T_Bus', 'Motorcycle': 'M_Cycle', 'Combination_Truck': 'C_Truck',
        'EV': 'E_Car', 'Light_Commercial_Truck': 'LC_Truck', 'Motor_Home': 'M_Home', 'Refuse_Truck': 'R_Truck',
        'School_Bus': 'S_Bus', 'Single_Unit_Truck': 'SU_Truck'}
# confusion_matrix = confusion_matrix.loc[confusion_matrix.index.isin(ndd), confusion_matrix.columns.isin(ndd)]
confusion_matrix = confusion_matrix.reset_index()
confusion_matrix['index'] = confusion_matrix['index'].replace(redd)
confusion_matrix.set_index('index', inplace=True)
confusion_matrix.rename(columns=redd, inplace=True)

confusion_matrixn = confusion_matrix.values.astype('float') / confusion_matrix.values.sum(axis=1)[:, np.newaxis]
confusion_matrixn = pd.DataFrame(confusion_matrixn)
confusion_matrixn.columns = confusion_matrix.columns
confusion_matrixn.index = confusion_matrix.index

fig, ax = plt.subplots(1, 1, figsize=(9.5, 6))
ax.grid(False)
sns.heatmap(confusion_matrixn * 100, annot=True, annot_kws={"fontsize": 14}, fmt=".1f", linewidth=.5, cmap='coolwarm',
            ax=ax, alpha=0.8)
plt.tight_layout()
plt.xticks(rotation=45)
plt.ylabel('')
plt.savefig(r'D:\NY_Emission\Figure\confusion_matrix_nc.pdf', dpi=600)
need_metrics = pd.DataFrame(confusion_matrix.da.export_metrics())
need_metrics.to_csv(r'D:\NY_Emission\Figure\confusion_matrix_nc.csv')

# Make prediction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learn = fva.load_learner(r'D:\NY_Emission\Cartype\f_model\20240508210959-%s\%s.pkl' % (model_nm, model_nm), cpu=False)
all_imgs = glob.glob(r'D:\NY_Emission\Cartype\MIO-TCD-Classification\train\bus\*')
for kk in all_imgs:
    sample_img = Image.open(kk)
    plt.imshow(sample_img)
    x = transform_single(sample_img, input_size=224, crop_pct=1)
    x.unsqueeze_(0)
    output = learn.model(x.to(device))
    output_top5 = output.topk(5)[1].cpu().numpy()[0]
    probabilities = F.softmax(output.topk(5).values, dim=1).tolist()[0]
    print("Truth: %s, Predicted: %s (%s)" % ('??', learn.dls.vocab[output_top5[0]], probabilities[0]))

# Ground truth of vehicle type
# If want to compare node-wise vehicle type, refer to D:\NY_Emission\Shp\AADT.shp
# ACT_AVG_MC_PERC: Motorcycles; ACT_AVG_CAR_PERC: Passenger_Car + EV; ACT_AVG_LT_PERC: Passenger_Truck + Light_Commercial_Truck
# ACT_AVG_BUS_PERC: Transit_Bus+Intercity_Bus+School_Bus; AVG_WKDAY_F5_7: Single_Unit_Truck+Motor_Home; ACT_AVG_CU_PERC: Combination_Truck+Refuse_Truck
vehicle_type_gr = pd.read_excel(r'D:\NY_Emission\Volume_grth\Count_Statistics_2022.xlsx')
vehicle_type_gr = vehicle_type_gr[vehicle_type_gr['COUNTY_FIPS'] == 61]  # New York County
vehicle_type_gr['MC_Count'] = vehicle_type_gr['AADT'] * vehicle_type_gr['ACT_AVG_MC_PERC']
vehicle_type_gr['CAR_Count'] = vehicle_type_gr['AADT'] * vehicle_type_gr['ACT_AVG_CAR_PERC']
vehicle_type_gr['LT_Count'] = vehicle_type_gr['AADT'] * vehicle_type_gr['ACT_AVG_LT_PERC']
vehicle_type_gr['BUS_Count'] = vehicle_type_gr['AADT'] * vehicle_type_gr['ACT_AVG_BUS_PERC']
vehicle_type_gr['SU_Count'] = vehicle_type_gr['AADT'] * vehicle_type_gr['AVG_WKDAY_F5_7']
vehicle_type_gr['CU_Count'] = vehicle_type_gr['AADT'] * vehicle_type_gr['ACT_AVG_CU_PERC']
tp_lt = ['MC_Count', 'CAR_Count', 'LT_Count', 'BUS_Count', 'SU_Count', 'CU_Count']
vehicle_type_gr[tp_lt] = vehicle_type_gr[tp_lt].fillna(0)
vehicle_type_gr[tp_lt].sum() / vehicle_type_gr[tp_lt].sum().sum()

# Get car type distribution from camera
count_df = pd.read_pickle(r'D:\NY_Emission\Video_Process\data_models\count_df.pkl')
p_c = ['c3_car_s', 'c3_truck_s', 'c3_bus_s', 'c3_cycle_s', 'c3_total_s']
ct_weight = count_df.groupby(['file', 'Hour'])['frame'].count() / 3600
ct_each = count_df.groupby(['file', 'Hour'])[p_c].sum().div(ct_weight, axis=0).reset_index()
ct_each['id'] = ct_each['file'].str.replace('.mp4', '')
ct_each[['c3_car_s', 'c3_truck_s', 'c3_bus_s', 'c3_cycle_s']].sum() / ct_each['c3_total_s'].sum()

# Our new model
count_df_new = pd.read_pickle(r'D:\NY_Emission\Video_Process\count_df_new_final.pkl')
p_c = ['c3_car_s', 'c3_ptruck_s', 'c3_htruck_s', 'c3_ltruck_s', 'c3_sbus_s', 'c3_bus_s', 'c3_mcycle_s']
ct_weight = count_df_new.groupby(['file', 'Hour'])['frame'].count() / 3600
ct_each = count_df_new.groupby(['file', 'Hour'])[p_c].sum().div(ct_weight, axis=0).reset_index()
ct_each['id'] = ct_each['file'].str.replace('.mp4', '')
ct_each['sum'] = ct_each[p_c].sum(axis=1)
temp = pd.DataFrame(ct_each[p_c].sum() / ct_each['sum'].sum()).reset_index()

# barplot
cardis = pd.read_excel(r'D:\NY_Emission\Figure\cardis.xlsx')
cardis.columns = ['Type', 'YOLO+Classifier', 'Truth', 'YOLO']
cardis = pd.melt(cardis, id_vars='Type', value_vars=['YOLO+Classifier', 'Truth', 'YOLO'])
# cardis.set_index(['Type'], inplace=True)
fig, ax = plt.subplots(figsize=(5, 5))
sns.barplot(cardis, y='Type', x='value', hue='variable', palette='tab20', ax=ax)
plt.ylabel('')
plt.legend(title='')
plt.xlabel('Fleet composition (%)')
plt.tight_layout()
plt.savefig(r'D:\NY_Emission\Figure\cartype_dis.pdf')


# Plot 12 types
def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h, w])

    # Centralize and crop
    crop_img = img[int(h / 2 - min_size / 2):int(h / 2 + min_size / 2),
               int(w / 2 - min_size / 2):int(w / 2 + min_size / 2)]
    resized = cv2.resize(crop_img, (int(size * 1.3), size), interpolation=interpolation)

    return resized


fig = plt.figure(figsize=(6.5, 4.5))
columns = 4
rows = 3
all_files = glob.glob(r'D:\NY_Emission\Cartype\carplot\*')
for i in range(1, columns * rows + 1):
    img = cv2.imread(all_files[i - 1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_square(img, 120)
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.title(all_files[i - 1].split('\\')[-1].split('.jpg')[0], fontsize=14)
    plt.imshow(img)
plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.05, left=0.035, right=0.965, hspace=0.232, wspace=0.082)
plt.savefig(r'D:\NY_Emission\Figure\cartype.pdf')
