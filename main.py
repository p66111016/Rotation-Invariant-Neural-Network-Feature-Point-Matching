import numpy as np
import cv2
import yaml
import os
import sys
import math

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from components import load_component
from utils import evaluation_utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='configs/super_sift_sg.yaml', # 'configs/super_sift_sg.yaml' or 'configs/sg_splight.yaml'
  help='number of processes.')
parser.add_argument('--img1_path', type=str, default= 'demo/demo_image/buiding_demo1.jpg',  #  'demo/demo_image/buiding_demo1.jpg'  or 'demo/demo_image/uav_demo1.jpg'
  help='number of processes.')
parser.add_argument('--img2_path', type=str, default= 'demo/demo_image/buiding_demo2.jpg',  #  'demo/demo_image/buiding_demo2.jpg'  or 'demo/demo_image/uav_demo2.jpg'
  help='number of processes.')
parser.add_argument('--rotation', type=str, default= 0,
  help='number of processes.')


args = parser.parse_args()


def rotate_image(image, angle):
    # Get the image dimensions
    height, width = image.shape[:2]

    # print(height, width)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Calculate the new dimensions of the rotated image
    new_width = int(width * abs(math.sin(math.radians(angle))) + height * abs(math.cos(math.radians(angle))))
    new_height = int(height * abs(math.sin(math.radians(angle)))+ width * abs(math.cos(math.radians(angle))))

    rotation_matrix[0, 2] += (new_width - width) / 2 
    rotation_matrix[1, 2] += (new_width - height) / 2

    # print(new_width, new_height)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    
    return rotated_image
    

if __name__=='__main__':
    with open(args.config_path, 'r', encoding='utf-8') as f:
      demo_config = yaml.load(f,Loader = yaml.FullLoader)

    extractor=load_component('extractor',demo_config['extractor']['name'],demo_config['extractor'])

    img1=cv2.imread(args.img1_path)
    size1=np.flip(np.asarray(img1.shape[:2]))
    kpt1,desc1=extractor.run(args.img1_path)
    
    matcher=load_component('matcher',demo_config['matcher']['name'],demo_config['matcher'])

    rotation_angle = args.rotation # Set the rotation angle in degrees

    img2 = cv2.imread(args.img2_path)
    rotation_angle = float(rotation_angle)
    
    rotated_img = rotate_image(img2, rotation_angle)
    
    size2=np.flip(np.asarray(rotated_img.shape[:2]))

    rotated_img = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)

    kpt2,desc2=extractor.run(rotated_img)
    
    test_data={'x1':kpt1,'x2':kpt2,'desc1':desc1,'desc2':desc2,'size1':size1,'size2':size2}
    corr1,corr2= matcher.run(test_data)

    print(f"Rotation angle: {rotation_angle} degrees")
    print(f"Number of matched points: {len(corr1)}")
    
    # draw points
    dis_points_1 = evaluation_utils.draw_points(img1, corr1)
    dis_points_2 = evaluation_utils.draw_points(rotated_img, corr2)
    
    saved_path = 'demo/result'  
    if not os.path.exists(saved_path):
          os.mkdir(saved_path)

    #visualize match
    display=evaluation_utils.draw_match(dis_points_1,dis_points_2,corr1,corr2)
    cv2.imwrite(os.path.join(saved_path,"match.png"), display)

        