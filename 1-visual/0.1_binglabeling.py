from ultralytics import YOLO
import cv2
import torch
import glob
import splitfolders
from bing_image_downloader import downloader
import os
from PIL import Image
from tqdm import tqdm
import hashlib
import os

# Download data from bing image
for kk in ["sedan", "coupe", "hatchback", "SUV", "crossover", "convertible", "minivan", "compact cars", "sport cars",
           "Pickup", "van", "passenger truck", "delivery van", "motorcycle", "scooter", "Coach bus", "School bus",
           "Transit Bus", "Tranist Bus new york", "Refuse Truck", "MotorHome", "Box Truck", "heavy truck", "dump truck",
           "truck trailer", "Light Cargo Truck", "Tesla Model Y", "Tesla Model 3", "Tesla", "Tesla Model S",
           "Tesla Model X", "VW ID.4", "Rivian R1S", "Rivian R1T", "Polestar 2", "Nissan Ariya", "Nissan Leaf",
           "Hyundai Ioniq", "Ford Mustang Mach-E", "Chevy Bolt EV", "BMW i4", "BMW iX", "Ford F-150 Lightning",
           "Mercedes EQ", "Kia EV6", "Audi Q4 e-tron", "Station wagon", "jeep", "ambulance", "Triple trailer truck",
           "Long combination truck", "Double trailer truck", "Semi Trailer", "Semi Truck Trailer", "Single-Unit Truck",
           "Class 4 Truck", "delivery truck", "bucket truck", "beverage truck", "garbage truck", "big rig",
           "cement truck", "used cars", "toyota sedan", "honda sedan", "hyundai sedan", "kia sedan", "ford sedan",
           "lexus sedan", "volvo sedan", "subaru sedan", "genesis sedan", "mazda sedan", "toyota suv", "honda suv",
           "hyundai suv", "kia suv", "ford suv", "lexus suv", "volvo suv", "subaru suv", "genesis suv", "mazda suv",
           "Toyota Pickup", "Ford Pickup", "delivery van fedex", "delivery van uhaul", "delivery van dhl",
           "delivery van amazon", "delivery van usps", "delivery van ups"]:
    downloader.download(kk, limit=350, output_dir="Dataset")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_yolo = YOLO()

Passenger_Car = ["sedan", "coupe", "convertible", "compact cars", "sport cars", "ford car", "genesis", "honda car",
                 "jeep", "toyota sedan", "honda sedan", "hyundai sedan", "kia sedan", "ford sedan",
                 "lexus sedan", "volvo sedan", "subaru sedan", "genesis sedan", "mazda sedan", "Station wagon"]
Passenger_Truck = ["SUV", "minivan", "Pickup", "van", "minibus_cv", "pickup_truck_cv", "van_cv"]
Light_Commercial_Truck = ["delivery van fedex", "delivery van uhaul", "delivery van dhl",
                          "delivery van amazon", "delivery van usps", "delivery van ups"]
Refuse_Truck = ["Refuse Truck", "garbage truck"]
Single_Unit_Truck = ["Box Truck", "Light Cargo Truck", "beverage truck", "Class 4 Truck", "Single-Unit Truck",
                     "delivery truck"]
Combination_Truck = ["heavy truck", "dump truck", "truck trailer", "trailer_truck_cv", "big rig",
                     "Double trailer truck", "Long combination truck", "Semi Truck Trailer", "Triple trailer truck"]
Intercity_Bus = ["Coach bus", "Intercity Coach"]
Transit_Bus = ["Transit Bus", "Tranist Bus new york"]
School_Bus = ["School bus"]
Motor_Home = ["MotorHome"]
EV = ["Tesla Model Y", "Tesla Model 3", "Tesla", "Tesla Model S", "Tesla Model X", "VW ID.4", "Rivian R1S",
      "Rivian R1T", "Polestar 2", "Nissan Ariya", "Nissan Leaf", "Hyundai Ioniq", "Ford Mustang Mach-E",
      "Chevy Bolt EV", "BMW i4", "BMW iX", "Ford F-150 Lightning", "Mercedes EQ", "Kia EV6", "Audi Q4 e-tron"]
Motorcycle = ['motorcycle', "motorbike_cv"]

# 0: 'person',1: 'bicycle',2: 'car',3: 'motorcycle',4: 'airplane', 5: 'bus',6: 'train',7: 'truck',
cc = 0
for kk in Passenger_Truck:
    allf = glob.glob(r"D:\NY_Emission\Cartype\Dataset\\" + kk + "\*")
    for rr in allf:
        image = cv2.imread(rr)
        if image is not None:
            results = model_yolo(image, conf=0.6, device=device)
            # Extract bounding boxes
            boxes = results[0].boxes.xyxy.tolist()
            types = results[0].boxes.cls.tolist()
            if len(boxes) > 0:
                # Iterate through the bounding boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    if (types[i] in [2, 7]) & (
                            (abs((x2 - x1) * (y2 - y1)) / (image.shape[0] * image.shape[1])) > 0.4):
                        # Crop the object using the bounding box coordinates
                        crop_object = image[int(y1):int(y2), int(x1):int(x2)]
                        # Save the cropped object as an image
                        cv2.imwrite(r'D:\NY_Emission\Cartype\\Final\\Passenger_Truck\\' + kk + str(cc) + '.jpg',
                                    crop_object)
                        cc += 1

# Remove small images
img_dirs = glob.glob(r"D:\NY_Emission\Cartype\MIO_New\\*")
for img_dir in tqdm(img_dirs):
    for filename in os.listdir(img_dir):
        filepath = os.path.join(img_dir, filename)
        with Image.open(filepath) as im:
            x, y = im.size
        if x < 128 or y < 128:
            os.remove(filepath)
        # else:
        #     image = cv2.imread(filepath)
        #     cv2.imshow("Input", image)
        #     results = model_yolo(image, conf=0.2, device=device)

# Remove similar images
hashes = set()
img_dirs = glob.glob(r"D:\NY_Emission\Cartype\Bing_label\Final\\*")
for img_dir in tqdm(img_dirs):
    for filename in os.listdir(img_dir):
        path = os.path.join(img_dir, filename)
        digest = hashlib.sha1(open(path, 'rb').read()).digest()
        if digest not in hashes:
            hashes.add(digest)
        else:
            os.remove(path)

# Split the folder
splitfolders.ratio(r'D:\NY_Emission\Cartype\Bing_label\Final', output=r"D:\NY_Emission\Cartype\Bing_label\Final_TVT",
                   seed=1337, ratio=(.8, 0.2, 0))
