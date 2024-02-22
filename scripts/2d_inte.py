# Standard libraries
import os
import random
import argparse
import glob

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
        description="This code integrates the segmentation results of the cropped image."
    )
    
    parser.add_argument("start", type=int, help="starting image number")
    parser.add_argument("end", type=int, help="ending image number")

    parser.add_argument(
        "--input", 
        type=str, 
        default="/mmdetection/petal_ct_crop_seg/data/crop_ct_seg", 
        help="Directory containing the segmentation results of the cropped image. Default is /mmdetection/petal_ct_crop_seg/data/crop_ct_seg")
    parser.add_argument(
        "--output", 
        type=str, 
        default="/mmdetection/petal_ct_crop_seg/data/2d_inte/", 
        help="Directory path to output results. Default is /mmdetection/petal_ct_crop_seg/data/2d_inte/")
    parser.add_argument(
        "--proc-num", 
        type=int, 
        default=multiprocessing.cpu_count(), 
        help="Number of cores used. Default is the maximum number of cores that can be used.")
    
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

# Function to find the bounding box of an object in a binary image
def find_coordinates(binary_image):
    # Find the coordinates of the target pixel in the image (for example, a pixel with a value of 1)
    coordinates = np.argwhere(binary_image > 0)
    
    # Calculate the upper left (minimum) and lower right (maximum) coordinates
    top_left = coordinates.min(axis=0)
    bottom_right = coordinates.max(axis=0)
    
    # Extract the coordinates
    top_left_y, top_left_x = top_left
    bottom_right_y, bottom_right_x = bottom_right

    # Extract the coordinates
    #width = bottom_right_x - top_left_x
    #height = bottom_right_y - top_left_y

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y

def cluster_inte(img_group_num):
    args = parse_args()
    
    # Reads a file list from the directory of crop images output from the trained model
    DIR_INPUT = os.path.join(args.input, f"ORA-{img_group_num:03d}")
    dir_name = os.path.splitext(os.path.basename(DIR_INPUT))[0]
    img_files = glob.glob(os.path.join(DIR_INPUT, "*"))
    img_files.sort()
    
    # Retrieve image information by extracting only one image
    img = cv2.imread(img_files[0])
    height, width, _ = img.shape
    
    # Count the number of clusters
    cluster_cnt = 0
    # All images are merged and stored by cluster number
    img_stack = np.zeros((360, height, width),dtype="u2")

    
    with tqdm(range(360)) as pbar:
        for angle in pbar:
            img_path = img_files[angle]
            img_name  = os.path.splitext(os.path.basename(img_path))[0]
            pbar.set_description(f"[{img_name}　Read]")
            
            cluster = generate_unique_color_mask_opencv(img_path)
            cluster = cluster + cluster_cnt
            img_stack[angle] = np.where(cluster > cluster_cnt, cluster, 0)
    
            cluster_cnt = cluster.max()
    
    crop_area_stack = np.zeros((360, height, width), dtype="?")
    
    # crop size
    crop_width = 900
    crop_height = 32
    
    # center coordinates
    obj_center_x = 421
    obj_center_y = 435

    # rotation angle
    angle_step = 1

    # Centering process
    center = (width // 2, height // 2)
    center_x_rotate, center_y_rotate = width // 2 - obj_center_x, height // 2 - obj_center_y

    # Calculate the upper left coordinate in the crop image
    rect_x = center[0] - crop_width // 2
    rect_y = center[1] - crop_height // 2

    # Randomly set colors
    colors = np.array([[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(8000)])
    
    # Affine transformation: translate
    tx, ty = -center_x_rotate, -center_y_rotate
    mv_mat = np.float32([[1, 0, tx],[0, 1, ty]])

    # Stores the cropping area
    with tqdm(range(0, 360, angle_step)) as pbar:
        for angle in pbar:
            pbar.set_description(f"[Setting {angle}°]")
    
            crop_area = np.zeros((height, width),dtype="u1")
            
            #　Rotate
            rotate_trans = cv2.getRotationMatrix2D(center, -angle , 1.0)
            rotate_temp  = cv2.warpAffine(crop_area, rotate_trans, (width, height))
    
            rotate_temp[rect_y:rect_y + crop_height, rect_x:rect_x + crop_width] = 255
    
            rotate_rev_trans = cv2.getRotationMatrix2D(center, angle , 1.0)
            rotate_rev_temp  = cv2.warpAffine(rotate_temp, rotate_trans, (width, height))
            rotate_rev_temp  = cv2.warpAffine(rotate_rev_temp, mv_mat, (width, height))
    
            crop_area_stack[angle] = rotate_rev_temp > 0

    # Remove incorrect segmentation results.
    # Retain more segmentation results.
    with tqdm(range(360)) as pbar:
        for source_img_num in pbar:
            img_name  = os.path.splitext(os.path.basename(img_files[source_img_num]))[0]
            pbar.set_description(f"[Removal {img_name}]")
    
            # Acquisition of cluster numbers in the source image
            source_cluster_list = [x for x in (np.unique(img_stack[source_img_num])).tolist() if x > 0]
            
            # Processing for each cluster number
            for source_cluster_num in source_cluster_list:
                
                # Reads out an array of source images
                source_cluster = img_stack[source_img_num]

                # Extract only the specified cluster number in source image
                source_comparison = source_cluster == source_cluster_num

                # Obtain the coordinates of the bounding box of the specified cluster
                source_th = (source_comparison * 255).astype(np.uint8)
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = find_coordinates(source_th)

                # Extract only the bounding box area
                img_part_stack = img_stack[:,top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                
                source_part_cluster = img_part_stack[source_img_num]
                source_part_comparison = source_part_cluster == source_cluster_num
                source_part_true_cnt = np.sum(source_part_comparison)
    
                #temp_ct_part_stack = crop_area_stack[:,top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                
                for target_img_num in range(360):

                    # Reads out an array of target images
                    target_part_cluster = img_part_stack[target_img_num]
                    temp = np.where(source_part_comparison, target_part_cluster, 0)
                    
                    if(np.sum(temp) > 0):
                        
                        # Acquisition of cluster numbers in the target image
                        target_cluster_list = [x for x in (np.unique(temp)).tolist() if x > 0]
                        
                        overlap_score = []
                        
                        if(len(target_cluster_list)>0):
    
                            #calc_and_overlap_cnt = np.sum(np.logical_and(source_part_comparison, np.logical_and(temp_ct_part_stack[source_img_num], temp_ct_part_stack[target_img_num])))
                            
                            for target_cluster_num in target_cluster_list:
                                target_part_comparison = temp == target_cluster_num
                                true_cnt_target = np.sum(target_part_comparison)
                    
                                # Calculate the number of True matches in the two numpy arrays
                                # Calculate True duplicates using logical conjunction (AND operation)
                                true_overlap = np.sum(np.logical_and(source_part_comparison, target_part_comparison))  
                                
                                # Calculate the percentage of True matches
                                true_overlap_ratio = true_overlap / source_part_true_cnt#calc_and_overlap_cnt
                                overlap_score.append(true_overlap_ratio)

                            # Extract elements with IoU greater than 0.05 and their indices
                            overlap_indices = [index for index, element in enumerate(overlap_score) if element > 0.05]

                            # If two or more petals are detected in the target cluster, the source cluster is removed
                            if(len(overlap_indices)>1):
                                img_stack[img_stack==source_cluster_num] = 0

    # Array to check for cluster number changes
    cluster_check_array = np.arange(cluster_cnt+1, dtype="u2")

    # List to store groups after integration
    cluster_list = [[] for i in range(cluster_cnt+1)]

    # Count of cluster numbers to be newly assigned
    new_cluster_id = 1

    # Integration using IoU 
    with tqdm(range(360)) as pbar:
        for source_img_num in pbar:
            img_name  = os.path.splitext(os.path.basename(img_files[source_img_num]))[0]
            pbar.set_description(f"[List-Integration {source_img_num}]")
    
            # Acquisition of cluster numbers in the source image
            source_cluster_list = [x for x in (np.unique(img_stack[source_img_num])).tolist() if x > 0]
            
            # Processing for each cluster number
            for source_cluster_num in source_cluster_list:
                
                flag = False
    
                overlap_list = []
                change_list = []
                if(cluster_check_array[source_cluster_num] != source_cluster_num):
                    change_list.append(cluster_check_array[source_cluster_num])
                
                overlap_list.append(source_cluster_num)
                
                source_cluster = img_stack[source_img_num]
                source_comparison = source_cluster == source_cluster_num
                source_true_cnt = np.sum(source_comparison)
    
                source_th = (source_comparison * 255).astype(np.uint8)
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = find_coordinates(source_th)
    
                img_part_stack = img_stack[:,top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                source_part_cluster = img_part_stack[source_img_num]
                source_part_comparison = source_part_cluster == source_cluster_num
                source_part_true_cnt = np.sum(source_part_comparison)
    
                temp_ct_part_stack = crop_area_stack[:,top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                
                for target_img_num in range(360):
                    
                    target_part_cluster = img_part_stack[target_img_num]
                    temp = np.where(source_part_comparison, target_part_cluster, 0)
                    
                    if(np.sum(temp) > 0):
                        
                        target_cluster_list = [x for x in (np.unique(temp)).tolist() if x > 0]
                        
                        overlap_score = []
    
                        calc_and_overlap_cnt = np.sum(np.logical_and(source_part_comparison, np.logical_and(temp_ct_part_stack[source_img_num], temp_ct_part_stack[target_img_num])))
                        
                        if(len(target_cluster_list)>0 and calc_and_overlap_cnt > int(source_part_true_cnt*0.7)):
                            
                            for target_cluster_num in target_cluster_list:
                                target_part_comparison = temp == target_cluster_num
                                true_cnt_target = np.sum(target_part_comparison)
                    
                                # Calculate the number of True matches in the two numpy arrays
                                # Calculate True duplicates using logical conjunction (AND operation)
                                true_overlap = np.sum(np.logical_and(source_part_comparison, target_part_comparison))
                                calc_and_overlap_cnt = np.sum(np.logical_and(np.logical_or(source_part_comparison, target_part_comparison), np.logical_and(temp_ct_part_stack[source_img_num], temp_ct_part_stack[target_img_num])))

                                if(calc_and_overlap_cnt>30):
                                
                                    # Calculate the percentage of True matches
                                    true_overlap_ratio = true_overlap / calc_and_overlap_cnt #source_part_true_cnt#
                                    overlap_score.append(true_overlap_ratio)
                        
                        overlap_indices = [index for index, element in enumerate(overlap_score) if element > 0.7]
                            
                        if(len(overlap_indices)>1):
                            cluster_check_array[source_cluster_num] = 0
                            flag = False
                            break
                        elif(len(overlap_indices)==1):
                            if(cluster_check_array[target_cluster_num] != target_cluster_num):
                                change_list.append(cluster_check_array[target_cluster_list[overlap_indices[0]]])
                            overlap_list.append(target_cluster_list[overlap_indices[0]])
                            flag = True
                        else:
                            flag = True
                            #break
                
                if(cluster_check_array[source_cluster_num] != 0 and flag == True):
                    if(len(set(change_list))==0):
                        inte_cluster_list = sorted(set(overlap_list))
                        inte_cluster_num  = inte_cluster_list[0]
    
                        cluster_list[inte_cluster_num] = inte_cluster_list
                        cluster_check_array[np.array(inte_cluster_list)] = inte_cluster_num
                        
                    elif(len(set(change_list))==1):
                        overlap_list.extend(cluster_list[list(set(change_list))[0]])
                        inte_cluster_list = sorted(set(overlap_list))
                        inte_cluster_num  = inte_cluster_list[0]
    
                        cluster_list[inte_cluster_num] = inte_cluster_list
    
                        for clear_c in inte_cluster_list:
                            if(clear_c != inte_cluster_num):
                                cluster_list[clear_c].clear()
    
                        cluster_check_array[np.array(inte_cluster_list)] = inte_cluster_num
                                
                    elif(len(set(change_list))>1):
                        change_list_fix = list(set(change_list))
                        for change_c in change_list_fix:
                            overlap_list.extend(cluster_list[change_c])
    
                        inte_cluster_list = sorted(set(overlap_list))
                        inte_cluster_num  = inte_cluster_list[0]
    
                        cluster_list[inte_cluster_num] = inte_cluster_list
    
                        for clear_c in inte_cluster_list:
                            if(clear_c != inte_cluster_num):
                                cluster_list[clear_c].clear()
    
                        cluster_check_array[np.array(inte_cluster_list)] = inte_cluster_num

    
    # Remove empty list
    filtered_list = [sublist for sublist in cluster_list if sublist]

    # Counting of newly assigned petal numbers
    new_cluster = 1

    inte_result_cluster = np.zeros(cluster_cnt+1, dtype="u2")
    inte_stack = np.zeros(img_stack.shape, dtype="u2")

    # Change of cluster number
    with tqdm(filtered_list) as pbar:
        for new_cluster_list in pbar:

            pbar.set_description(f"[Change cluster {dir_name}]")
            for cluster_num in new_cluster_list:
                inte_result_cluster[cluster_num] = new_cluster
            new_cluster += 1

    # Create newly numbered sequences
    with tqdm(range(360)) as pbar:
        for img_num in pbar:
            
            pbar.set_description(f"[Assign new-cluster {dir_name}]")
            for y, x in np.argwhere(img_stack[img_num] > 0):
                inte_stack[img_num,y,x] = inte_result_cluster[int(img_stack[img_num,y,x])]
    
    # Array to store integration results
    inte_cluster_array = np.zeros((height, width), dtype="u2")
    
    mask = inte_stack != 0
    
    # Calculate mode for each pixel
    with tqdm(range(height)) as pbar:
        for y in pbar:
            for x in range(width):
                
                pbar.set_description(f"[Calculate mode {dir_name}]")

                # Get non-zero value
                temp = inte_stack[:, y, x][mask[:, y, x]]
        
                # Calculate mode if temp is not empty
                if temp.size > 0:
                    inte_cluster_array[y, x] = stats.mode(temp)[0]

    # Image array to store segmentation results
    inte_img = np.zeros((height,width,3),dtype="u1")

    with tqdm(range(height)) as pbar:
        for y in pbar:
            for x in range(width):
                
                pbar.set_description(f"[Assign color {dir_name}]")
                
                if(inte_cluster_array[y,x]>0):
                    inte_img[y,x] = colors[inte_cluster_array[y,x]]

    # Export segmentation results
    os.makedirs(args.output, exist_ok=True)
    DIR_OUTPUT = os.path.join(args.output, f"ORA-{img_group_num:03d}_inte.png")
    
    cv2.imwrite(DIR_OUTPUT, inte_img)

def process_images(args):
    proc_list = args
    
    for i in proc_list:
        cluster_inte(i)

def main():

    args = parse_args()

    args_func = []
    
    for i in range(args.proc_num):
        temp = list(range(args.start+i, args.end+1, args.proc_num))
        args_func.append(temp)
    
    with ProcessPoolExecutor() as executor:
        executor.map(process_images, args_func)

if __name__ == "__main__":
    main()