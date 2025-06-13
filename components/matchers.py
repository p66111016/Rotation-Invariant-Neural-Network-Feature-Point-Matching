import torch
import numpy as np
import cv2
import os
from collections import OrderedDict,namedtuple
import sys
import matplotlib.pyplot as plt
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from superglue import matcher as SG_Model
from superglue import pretrain_matcher as Sp_Sift_Sg_Model           #sift_tanh
from utils import evaluation_utils


class GNN_Matcher(object):

    def __init__(self,config,model_name):
        assert model_name=='sp_sift_sg' or model_name=='SG'

        config=namedtuple('config',config.keys())(*config.values())
        self.p_th=config.p_th
        self.model = Sp_Sift_Sg_Model(config) if model_name== 'sp_sift_sg' else SG_Model(config) 
        self.model.cuda(),self.model.eval()
        checkpoint = torch.load(os.path.join(config.model_dir, 'model_best.pth'))
        
        #for ddp model
        if list(checkpoint['state_dict'].items())[0][0].split('.')[0]=='module':
            new_stat_dict=OrderedDict()
            for key,value in checkpoint['state_dict'].items():
                new_stat_dict[key[7:]]=value
            checkpoint['state_dict']=new_stat_dict
        self.model.load_state_dict(checkpoint['state_dict'])

    def run(self,test_data):
        norm_x1,norm_x2=evaluation_utils.normalize_size(test_data['x1'][:,:2],test_data['size1']),\
                                                    evaluation_utils.normalize_size(test_data['x2'][:,:2],test_data['size2'])                                  
    
        x1,x2=np.concatenate([norm_x1,test_data['x1'][:,2,np.newaxis]],axis=-1),np.concatenate([norm_x2,test_data['x2'][:,2,np.newaxis]],axis=-1)
        
        feed_data={'x1':torch.from_numpy(x1[:,:2][np.newaxis]).cuda().float(),
                   'x2':torch.from_numpy(x2[:,:2][np.newaxis]).cuda().float(),
                   'desc1':torch.from_numpy(test_data['desc1'][np.newaxis]).cuda().float(),
                   'desc2':torch.from_numpy(test_data['desc2'][np.newaxis]).cuda().float(),
                   'pscore1':torch.from_numpy(test_data['x1'][:,2,np.newaxis][np.newaxis]).cuda().float(),
                   'pscore2':torch.from_numpy(test_data['x2'][:,2,np.newaxis][np.newaxis]).cuda().float()
                   }
        
        with torch.no_grad():
            res=self.model(feed_data,test_mode=True)
            p=res['p']
        index1,index2=self.match_p(p[0,:-1,:-1])
        corr1,corr2=test_data['x1'][:,:2][index1.cpu()],test_data['x2'][:,:2][index2.cpu()]
        
        if len(corr1.shape)==1:
            corr1,corr2=corr1[np.newaxis],corr2[np.newaxis]
        return corr1,corr2
    
    
    
    def match_p(self,p):#p N*M
        score,index=torch.topk(p,k=1,dim=-1)
        _,index2=torch.topk(p,k=1,dim=-2)
        mask_th,index,index2=score[:,0]>self.p_th,index[:,0],index2.squeeze(0)
        mask_mc=index2[index] == torch.arange(len(p)).cuda()
        mask=mask_th&mask_mc
        index1,index2=torch.nonzero(mask).squeeze(1),index[mask]
        return index1,index2

