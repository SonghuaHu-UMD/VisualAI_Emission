import cv2
import torch
import pandas as pd
import glob
import timm
import os
import fastai.vision.all as fva
from torch.nn import functional as F
import random
import dill
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# from torchtnt.utils import get_module_summary
import datetime
from pathlib import Path
import numpy as np
import seaborn as sns
# from fastai.vision.all import *
import disarray
import shutil
import splitfolders

# multiprocessing.set_start_method('spawn', force=True)

pd.options.mode.chained_assignment = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def transform_single(img, input_size, crop_pct=1, mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
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

# Split the folder for model training
splitfolders.ratio(r'D:\NY_Emission\Cartype\Crop_NY_Final', output=r"D:\NY_Emission\Cartype\Bing_label\Final_TVT",  seed=1337, ratio=(.8, 0.2, 0))

if __name__ == '__main__':
    # Load data
    r_sz = 224
    dls = fva.ImageDataLoaders.from_folder(r'D:\NY_Emission\Cartype\Bing_label\Final_TVT', train="train", valid="val",
                                           label_func=lambda x: x[0].isupper(), item_tfms=fva.Resize(r_sz), bs=32,
                                           num_workers=4)
    # batch_tfms = fva.aug_transforms(do_flip=True, flip_vert=False, mult=2.0), bs = 30, num_workers = 4
    # dls.show_batch()

    # Select a model from timm: https://paperswithcode.com/lib/timm
    # model_list = pd.DataFrame(timm.list_models('vit*', pretrained=True))
    for model_name in ['vit_small_patch16_224', 'swin_base_patch4_window7_224', 'convnext_tiny', 'repvgg_a2',
                       'inception_v4', 'resnet50', 'densenet201', 'inception_next_tiny', 'xception71',
                       'efficientnetv2_rw_t']:
        # Read model
        # model_name = 'efficientnetv2_rw_t'
        model_ti = timm.create_model(model_name, pretrained=True, num_classes=dls.c)
        data_config = timm.data.resolve_model_data_config(model_ti)
        parameter_size = sum(p.numel() for p in model_ti.parameters() if p.requires_grad) / 1e6
        print("# Parameter of %s: %sM" % (model_name, str(parameter_size)))
        # model_ti = model_ti.to(device=device)
        # model_ti.default_cfg
        # get_module_summary(model_ti)

        # Generate path for the training session
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '-' + model_name
        checkpoint_dir = Path(r'D:\NY_Emission\Cartype\model\\%s' % timestamp)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Fine tune the model
        f1score_ins = fva.F1Score(average='weighted')
        Precision_ins = fva.Precision(average='weighted')
        Recall_ins = fva.Recall(average='weighted')
        Focal_ins = fva.FocalLossFlat(reduction='mean')  # loss_func=Focal_ins,
        learn = fva.Learner(dls, model_ti, loss_func=Focal_ins,
                            metrics=[fva.accuracy, fva.top_k_accuracy, Precision_ins, Recall_ins, f1score_ins],
                            model_dir=checkpoint_dir.absolute())
        suggested_lrs = learn.lr_find()
        lr = suggested_lrs.valley * 4
        start = datetime.datetime.now()
        learn.fine_tune(epochs=15, base_lr=lr, cbs=[fva.MixedPrecision(), fva.SaveModelCallback(),
                                                    fva.EarlyStoppingCallback(min_delta=1e-3, patience=5)])
        end = datetime.datetime.now() - start

        # Best model
        learn.load_state_dict(torch.load(r'D:\NY_Emission\Cartype\model\%s\model.pth' % timestamp))

        # torch.save(learn.model.state_dict(), str(checkpoint_dir.absolute()) + "\\%s.pth" % model_name)
        learn.export(str(checkpoint_dir.absolute()) + "\\%s.pkl" % model_name, pickle_module=dill)

        # Get the train and val loss
        # learn.recorder.plot_loss()
        metrics_val = pd.DataFrame(learn.recorder.values)
        metrics_val.columns = ['train_loss', 'valid_loss', 'accuracy', 'top_k_accuracy', 'precision_score',
                               'recall_score', 'f1_score']
        metrics_train = pd.DataFrame(learn.recorder.losses).astype("float")
        metrics_train.columns = ['loss']
        metrics_train['para'] = parameter_size
        metrics_train['time'] = end

        # Get the confusion_matrix
        interp = fva.ClassificationInterpretation.from_learner(learn)
        # interp.plot_confusion_matrix(figsize=(12, 12), normalize=True, cmap='coolwarm')
        cm = interp.confusion_matrix()
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = pd.DataFrame(cm)
        cm.columns = interp.vocab
        cm.index = interp.vocab
        cm = cm.astype(int)
        # sns.heatmap(cm.div(cm.max(axis=1), axis=0))
        need_metrics = pd.DataFrame(cm.da.export_metrics())
        # interp.plot_top_losses(k=12, figsize=(15,11))
        # interp.most_confused(min_val=3)

        # Output the model
        cm.to_pickle(str(checkpoint_dir.absolute()) + "\\confusion_matrix.pkl")
        metrics_val.to_pickle(str(checkpoint_dir.absolute()) + "\\metrics_val.pkl")
        metrics_train.to_pickle(str(checkpoint_dir.absolute()) + "\\metrics_train.pkl")

        # # # # Test the model prediction
        # item_path = random.choice(dls.valid_ds.items)
        # sample_img = Image.open(item_path)
        # plt.imshow(sample_img)
        # x = transform_single(sample_img, input_size=r_sz, crop_pct=1)
        # x.unsqueeze_(0)
        # output = learn.model(x.to(device))
        # output_top5 = output.topk(5)[1].cpu().numpy()[0]
        # probabilities = F.softmax(output.topk(5).values, dim=1).tolist()[0]
        # print("Truth: %s, Predicted: %s (%s)" % (item_path.parent.name, dls.vocab[output_top5[0]], probabilities[0]))
