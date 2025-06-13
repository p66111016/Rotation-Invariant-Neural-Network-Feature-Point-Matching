import cv2
import numpy as np
import torch
import os

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from superpoint import SuperPoint_Light


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.tensor(image / 255., dtype=torch.float)


def resize(img,resize):
    img_h,img_w=img.shape[0],img.shape[1]
    cur_size=max(img_h,img_w)
    if len(resize)==1: 
      scale1,scale2=resize[0]/cur_size,resize[0]/cur_size
    else:
      scale1,scale2=resize[0]/img_h,resize[1]/img_w
    new_h,new_w=int(img_h*scale1),int(img_w*scale2)
    new_img=cv2.resize(img.astype('float32'),(new_w,new_h)).astype('uint8')
    scale=np.asarray([scale2,scale1])
    return new_img,scale


class ExtractSuperpoint_Light(object):
  def __init__(self,config):
    default_config = {
      'detection_threshold': config['det_th'],
      'max_num_keypoints': config['num_kpt'],
      'remove_borders': 4,
      'model_path':'/mnt/g/SGMNet-original-sp_light_rgb/lightglue/weights/superpoint_v1.pth'
    }
    self.superpoint_extractor=SuperPoint_Light(default_config)
    self.superpoint_extractor.eval(),self.superpoint_extractor.cuda()
    self.num_kp=config['num_kpt']
    if 'padding' in config.keys():
      self.padding=config['padding']
    else:
      self.padding=False
    self.resize=config['resize']

  def run(self, img):
    # if type(img) == str:
    #   img = cv2.imread(img)
    #   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      
    img_torch = numpy_image_to_torch(img)
    
    scale=1
    if self.resize[0]!=-1:
      img,scale=resize(img,self.resize)

    # print("start extract")

    with torch.no_grad():
      result = self.superpoint_extractor.extract(img_torch.cuda())
      # result=self.superpoint_extractor(torch.from_numpy(img/255.).float()[None, None].cuda())
    
    # print("extract success")
    score,kpt,desc=result['keypoint_scores'][0],result['keypoints'][0],result['descriptors'][0]
    score,kpt,desc=score.cpu().numpy(),kpt.cpu().numpy(),desc.cpu().numpy()
    kpt=np.concatenate([kpt/scale,score[:,np.newaxis]],axis=-1)

    #padding randomly
    if self.padding:
      if len(kpt)<self.num_kp:
        res=int(self.num_kp-len(kpt))
        pad_x,pad_desc=np.random.uniform(size=[res,2])*(img.shape[0]+img.shape[1])/2,np.random.uniform(size=[res,256])
        pad_kpt,pad_desc=np.concatenate([pad_x,np.zeros([res,1])],axis=-1),pad_desc/np.linalg.norm(pad_desc,axis=-1)[:,np.newaxis]
        kpt,desc=np.concatenate([kpt,pad_kpt],axis=0),np.concatenate([desc,pad_desc],axis=0)
        
    return kpt,desc
  

class ExtractSuperSift(object):
  def __init__(self,config,root=True):
    default_config = {
      'descriptor_dim': 256,
      'nms_radius': 4,
      'detection_threshold': config['det_th'],
      'max_num_keypoints': config['num_kpt'],
      'remove_borders': 4,
      'model_path':'../lightglue/weights/superpoint_v1.pth'
    }
    self.root=root
    
    self.superpoint_extractor=SuperPoint_Light(default_config)
    self.superpoint_extractor.eval(),self.superpoint_extractor.cuda()
    self.num_kp=config['num_kpt']
    if 'padding' in config.keys():
      self.padding=config['padding']
    else:
      self.padding=False
    self.resize=config['resize']
    
    self.contrastThreshold = 0.00001

  def run(self,img_path):
    self.sift = cv2.SIFT_create(nfeatures=self.num_kp, contrastThreshold=self.contrastThreshold)
    if type(img_path) == str:
      img = cv2.imread(img_path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
      img = img_path
    
    scale=1
    if self.resize[0]!=-1:
      img,scale=resize(img,self.resize)


    img_torch = numpy_image_to_torch(img)
    
    with torch.no_grad():
      result = self.superpoint_extractor.extract(img_torch.cuda())
      # result=self.superpoint_extractor(torch.from_numpy(img/255.).float()[None, None].cuda())
  
    score,kpt=result['keypoint_scores'][0],result['keypoints'][0]
    score,kpt=score.cpu().numpy(),kpt.cpu().numpy()
    
    kp = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in kpt]
    if len(kp) == 0:
      self.root = False
    
    _, desc = self.sift.compute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), kp)
    
    kpt=np.concatenate([kpt/scale,score[:,np.newaxis]],axis=-1)
    
    
    if self.root:
      desc=np.sqrt(abs(desc/(np.linalg.norm(desc,axis=-1,ord=1)[:,np.newaxis]+1e-8)))
    
    #padding randomly
    if self.padding:
      if len(kpt)<self.num_kp:
        res=int(self.num_kp-len(kpt))
        pad_x,pad_desc=np.random.uniform(size=[res,2])*(img.shape[0]+img.shape[1])/2,np.random.uniform(size=[res,256])
        pad_kpt,pad_desc=np.concatenate([pad_x,np.zeros([res,1])],axis=-1),pad_desc/np.linalg.norm(pad_desc,axis=-1)[:,np.newaxis]
        kpt,desc=np.concatenate([kpt,pad_kpt],axis=0),np.concatenate([desc,pad_desc],axis=0)
    return kpt,desc
  
  
  

  