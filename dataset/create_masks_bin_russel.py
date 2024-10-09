"""
To create mask from polygon xml files
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

ANNOTATIONS_PATH = 'annotations/annotations.xml'
IMAGE_FOLDER_PATH = "Fetus/"
GT_FOLDER_PATH = "Fetus_GT/"



try:
    os.makedirs(GT_FOLDER_PATH)
except:
    pass



create_img_folder = "Fetus_IMG"

try:
    os.makedirs(create_img_folder)
except:
    pass


tree = ET.parse(ANNOTATIONS_PATH)
root = tree.getroot()

for child in root:
    if len(child.attrib) > 0:
        # print(child.tag, child.attrib['name'])
        img_name = child.attrib['name']
        img_path = IMAGE_FOLDER_PATH+img_name
        print("image path => ", img_path)
        img_path_act = "Fetus/"+img_name
        img_act = cv2.imread(img_path_act, cv2.IMREAD_UNCHANGED)
        # read the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # create zero matrix of the same size of the image
        mask = np.zeros(img.shape, dtype=np.uint8)
        for childs_child in child:
            # print(childs_child.tag, childs_child.attrib)
            if len(childs_child.attrib) > 0:
                points = childs_child.attrib['points']
                print(points)
                points = points.split(';')
                points = [point.split(',') for point in points]
                points = [[int(float(point[0])), int(float(point[1]))] for point in points]
                points = np.array(points, dtype=np.int32)

                # Draw polygon
                cv2.fillPoly(mask, [points], 255)


                 # Here GT is the convex hull of the pred_mask
                gt_mask = mask.copy()
                gt_mask = np.array(gt_mask, np.uint8)
                contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # print("contours = ",contours)
                convex_hulls = []
                for contour in contours:
                    convex_hull = cv2.convexHull(contour)
                    convex_hulls.append(convex_hull)
                cv2.drawContours(gt_mask, convex_hulls, -1, 1, 1)  # Draw green lines for the hulls
                
                # save the mask
                #mask_name = "masks_bin_jimut/"+str(img_name.split('.')[0])+".png"
                mask_name = GT_FOLDER_PATH+str(img_name.split('.')[0])+".png"
                cv2.imwrite(mask_name, gt_mask)

                img_name = "Fetus_IMG/"+str(img_name.split('.')[0])+".png"
                cv2.imwrite(img_name, img_act)
                
                #print("saved mask: ", mask_name, mask.shape)