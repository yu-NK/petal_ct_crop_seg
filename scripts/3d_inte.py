# Standard libraries
import os
import random
import argparse
import glob
import copy

import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor

# Third-party libraries
from PIL import Image, ImageDraw
import cv2
import numpy as np
from tqdm import tqdm
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(
        description="This code integrates segmentation results that are integrated in 2D in a 3D direction."
    )

    parser.add_argument(
        "--input", 
        type=str, 
        default="/mmdetection/petal_ct_crop_seg/data/2d_inte/", 
        help="Directory where the segmentation results merged in 2D are stored. Default is /mmdetection/petal_ct_crop_seg/data/2d_inte/")
    parser.add_argument(
        "--output", 
        type=str, 
        default="/mmdetection/petal_ct_crop_seg/data/3d_inte/", 
        help="Directory path to output results. Default is /mmdetection/petal_ct_crop_seg/data/3d_inte/")
    
    args = parser.parse_args()

    return args
    
# Function to assign a number for each color
def generate_unique_color_mask_opencv(image_path):
    # Using OpenCV to load images
    image = cv2.imread(image_path)
    
    # Create a unique set of colors
    unique_colors, unique_indices = np.unique(image.reshape(-1, 3), axis=0, return_inverse=True)
    
    # Create mask array with unique color indices
    mask_array = unique_indices.reshape(image.shape[:2])
    
    return mask_array #, {tuple(color): idx for idx, color in enumerate(unique_colors)}

def main():
    args = parse_args()
    
    # Obtain the path of the segmentation results merged in 2D
    img_files = glob.glob(os.path.join(args.input, "*"))
    img_files.sort()
    
    # Retrieve image information by extracting only one image
    img = cv2.imread(img_files[0])
    height, width, _ = img.shape

    # Randomly set colors
    colors = np.array([[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(20000)])
    
    # Variable to count cluster numbers
    cluster_cnt = 0
    # Array to store the cluster number of the 2d integration result
    img_stack = np.zeros((len(img_files), height,width),dtype="u2")

    #　Read 2D integration results
    with tqdm(range(len(img_files))) as pbar:
        for img_num in pbar:
            img_path = img_files[img_num]
            img_name  = os.path.splitext(os.path.basename(img_path))[0]
            pbar.set_description(f"[Read {img_name}]")
    
            mask_array = generate_unique_color_mask_opencv(img_path)
            cluster = mask_array + cluster_cnt
            img_stack[img_num] = np.where(cluster>cluster_cnt, cluster, 0)
            
            cluster_cnt = cluster.max()

    # 
    # Array to check for cluster number changes
    cluster_check_array = np.arange(cluster_cnt+1, dtype="u2")

    # List to store groups after integration
    cluster_list = [[] for i in range(cluster_cnt+1)]

    # Count of cluster numbers to be newly assigned
    new_cluster_id = 1
    
    new_cluster = copy.deepcopy(img_stack)

    # Integration using IoU 
    with tqdm(range(1,len(img_files))) as pbar:
        for source_img_num in pbar:
            img_name  = os.path.splitext(os.path.basename(img_files[source_img_num]))[0]
            pbar.set_description(f"[List-Integration {source_img_num}]")
    
            # Acquisition of cluster numbers in the source image
            source_cluster_list = [x for x in (np.unique(new_cluster[source_img_num])).tolist() if x > 0]
            
            # Processing for each cluster number
            for source_cluster_num in source_cluster_list:
                
                flag = False
    
                overlap_list = []
                
                source_cluster = new_cluster[source_img_num]
                source_comparison = source_cluster == source_cluster_num
                source_true_cnt = np.sum(source_comparison)
                
                for target_img_num in range(max(0, source_img_num - 2), source_img_num):
                    
                    target_cluster = new_cluster[target_img_num]
                    temp = np.where(source_comparison, target_cluster, 0)
                    
                    if(np.sum(temp) > 0 and target_img_num != source_img_num):
                        target_cluster_list = [x for x in (np.unique(temp)).tolist() if x > 0]
    
                        iou_score = []
                        
                        if(len(target_cluster_list)>0):
                            for target_cluster_num in target_cluster_list:
                                target_comparison = temp == target_cluster_num
                                true_cnt_target = np.sum(target_comparison)
                    
                                # Calculate the number of True matches in the two numpy arrays
                                # Calculate True duplicates using logical conjunction (AND operation)
                                true_overlap = np.sum(np.logical_and(source_comparison, target_comparison))
                                true_or_cnt = np.sum(np.logical_or(source_comparison, target_comparison))
                                
                                # Calculate the percentage of True matches
                                iou = true_overlap / true_or_cnt
                                iou_score.append(iou)
                                
                            iou_indices = [index for index, element in enumerate(iou_score) if element > 0]
    
                            if(len(iou_indices) > 0):
                                index = iou_score.index(max(iou_score))
                                overlap_list.append(target_cluster_list[index])
                                flag = True
                            else:
                                flag = True
    
                if(flag == True and len(overlap_list)>0):
                    inte_cluster_list = sorted(set(overlap_list))
                    inte_cluster_num  = inte_cluster_list[0]
    
                    source_cluster[source_comparison] = inte_cluster_num
                    new_cluster[source_img_num] = source_cluster

    with tqdm(range(len(img_files))) as pbar:
        for img_num in pbar: 
            inte_img = np.zeros((height,width,3),dtype="u1")
            inte_temp = new_cluster[img_num]

            pbar.set_description(f"[Assign color {img_num}]")
            
            for y in range(height):
                for x in range(width):
                    if(inte_temp[y,x] > 0):
                        inte_img[y,x] = colors[inte_temp[y,x]]
            
            os.makedirs(args.output, exist_ok=True)
            DIR_OUTPUT = os.path.join(args.output, f"ORA-{img_num:03d}.png")
            
            cv2.imwrite(DIR_OUTPUT, inte_img)

if __name__ == "__main__":
    main()