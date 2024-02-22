# Standard libraries
import os
import random
import argparse
import glob

# Third-party libraries
from PIL import Image, ImageDraw
import cv2
import numpy as np
from tqdm import tqdm

# Local application/library specific imports
from mmdet.apis import init_detector, inference_detector

def parse_args():
    parser = argparse.ArgumentParser(
        description="This code segments petals in cropped images using a trained model."
    )

    parser.add_argument('config', type=str, help='train config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file')
    parser.add_argument("start", type=int, help="starting image number")
    parser.add_argument("end", type=int, help="ending image number")
    
    parser.add_argument("--device", type=int, default=0, help="GPU device number. Dafault is 0.")
    parser.add_argument("--input", type=str, default="/mmdetection/petal_ct_crop_seg/data/volume_1", help="Directory path where images are stored. Default is /mmdetection/petal_ct_crop_seg/data/volume_1")
    parser.add_argument("--output", type=str, default="/mmdetection/petal_ct_crop_seg/crop_ct_seg", help="Directory path to output results. Default is /mmdetection/petal_ct_crop_seg/crop_ct_seg")
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # Check if input directory exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"The directory {args.input} does not exist.")

    # specifies the GPU device number
    device = "cuda:" + str(args.device)

    # crop size
    crop_width = 900
    crop_height = 32
    
    # center coordinates
    obj_center_x = 421
    obj_center_y = 435

    # rotation angle
    angle_step = 1

    # Randomly set colors
    colors = np.array([[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(1000)])

    # Set confidence score for model output
    score_conf = 0.98

    # Average value of background pixels
    bg_mean = 33

    for img_num in range(args.start, args.end + 1):
            
        img_path = os.path.join(args.input, f'ORA-{img_num:03d}.tif')
        img_name  = os.path.splitext(os.path.basename(img_path))[0]

        # Image loading
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # Noise Removal
        img_NLMD = cv2.fastNlMeansDenoising(img, h=6)

        # Numpy -> Pillow
        image = Image.fromarray(img_NLMD)

        # Centering process
        center_x, center_y = width // 2, height // 2
        center_x_rotate, center_y_rotate = width // 2 - obj_center_x, height // 2 - obj_center_y

        image_center = image.rotate(0, translate=(center_x_rotate, center_y_rotate))

        # Calculate the upper left coordinate in the crop image
        rect_x = center_x - crop_width // 2
        rect_y = center_y - crop_height // 2

        # Segmentation using trained models for each angle
        with tqdm(range(0, 360, angle_step)) as pbar:
            for angle in pbar:
                pbar.set_description(f"[{img_name} {angle}°]")
                
                cnt = 1
                
                # Rotate
                rotated_image = image_center.rotate(angle)
                rotated_image_RGB = rotated_image.convert('RGB')
                rotated_image_np = np.array(rotated_image_RGB, dtype="u1")
        
                # Crop from rotated image
                crop_image = rotated_image.crop((rect_x, rect_y, rect_x + crop_width, rect_y + crop_height))
                crop_image_np = cv2.cvtColor(np.array(crop_image, dtype="u1"), cv2.COLOR_BGR2GRAY)
        
                # Interpolate pixels lost due to rotation
                zero_positions = np.argwhere(crop_image_np == 0)
                random_values = np.random.randint(bg_mean-5, bg_mean+5, size=zero_positions.shape[0])
                crop_image_np[zero_positions[:, 0], zero_positions[:, 1]] = random_values
                crop_img_col = cv2.cvtColor(crop_image_np, cv2.COLOR_GRAY2BGR)
        
                # Input to trained model
                model = init_detector(args.config, args.checkpoint, device=device)
                result = inference_detector(model, crop_img_col)
                
                # Array to store results
                crop_result = np.zeros((crop_height,crop_width,3),dtype="u1")
        
                # Store segmentation results with color for each petal
                seg_cnt = 0
                
                if(len(result[1][0])>0):
                    for segm in result[1][0]:
                        if(result[0][0][seg_cnt][4] > score_conf):
                            segm_bin = np.zeros((crop_height, crop_width),dtype="u1")
                            
                            segm_bool = (np.array(segm[:, :, np.newaxis]))[:,:,0]
                            segm_bin[segm_bool] = 255
        
                            nLabels, labelImages, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(segm_bin, 8, cv2.CV_16U, cv2.CCL_DEFAULT)
        
                            if(nLabels>2):
                                for i in range(1,nLabels):
                                    label_bool = labelImages == i
                                    crop_result[label_bool] = cnt
                                    cnt += 1
        
                            else:
                                segm_rgb = np.repeat(segm[:, :, np.newaxis], 3, axis=2)
                                crop_result = np.where(segm_rgb==(True,True,True),colors[cnt],crop_result)
                                cnt += 1
        
                            seg_cnt += 1
        
                # Output results without background
                rotated_image_noback = np.zeros((height,width,3),dtype="u1")
                rotated_image_noback[rect_y:rect_y + crop_height, rect_x:rect_x + crop_width] = crop_result
                
                # Reverse rotation to revert to before rotation
                rotate_reverse = np.array((((Image.fromarray(rotated_image_noback)).convert('RGB')).rotate(-angle)).rotate(0, translate=(-center_x_rotate, -center_y_rotate)), dtype="u1")

                # Export segmentation results
                DIR_OUTPUT = os.path.join(args.output,f"ORA-{img_num:03d}/")
                os.makedirs(DIR_OUTPUT, exist_ok=True)

                file_OUTPUT = os.path.join(DIR_OUTPUT, (img_name + f"_{angle:03d}.png"))
                cv2.imwrite(file_OUTPUT, rotate_reverse)
        
if __name__ == "__main__":
    main()