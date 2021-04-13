import os
import glob
import torch
import random
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tools.misc import get_prob_map, get_thick_edge
from skimage import exposure
from skimage.transform import resize
from scipy import ndimage
from tools.transforms import *
import elasticdeform


def get_loaders(cf, phase='train'):
    #patients = os.listdir(cf.dataset_path)
    patients_train = os.listdir(cf.dataset_path+'train/')
    patients_val = os.listdir(cf.dataset_path+'validation/')
    patients_test = os.listdir(cf.dataset_path+'test/')
    
    #whether to shuffle or not the images
    if cf.shuffle_data:
        random.shuffle(patients_train)
        random.shuffle(patients_val)
        random.shuffle(patients_test)
        
###    #Creating Data Generator per split
    train_set = HeadyNeckDataset(patients_train, cf, 'train')
    val_set = HeadyNeckDataset(patients_val, cf, 'validation')
    test_set = HeadyNeckDataset(patients_test, cf, 'test')

    train_gen = DataLoader(train_set, batch_size=cf.batch_size)
    val_gen = DataLoader(val_set, batch_size=cf.batch_size)
    test_gen = DataLoader(test_set, batch_size=cf.batch_size)
    
    return train_gen, val_gen, test_gen


class HeadyNeckDataset(Dataset):
    """Heand and Neck Cancer dataset."""

    def __init__(self, indices, cf, phase):
        """
        Args:
            indices : list of the indices for this generator
            cf (Config file): set up info
            phase: train loader or eval loader. Important to apply or not DA.
        """
        self.indices = indices
        self.cf = cf
        self.phase = phase
        #self.transform = None
        self.transform = (self.cf.da_hor_flip or 
                          self.cf.da_ver_flip or 
                          self.cf.da_rotate or  
                          self.cf.da_deform)
        self.folder = self.phase + '/'

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx): 
        img_name = os.path.join(self.cf.dataset_path, self.folder, self.indices[idx], 'T1.npy')
        img_name2 = os.path.join(self.cf.dataset_path, self.folder, self.indices[idx], 'T1.npy')
        img_name1 = os.path.join(self.cf.dataset_path, self.folder, self.indices[idx], 'T1.npy')   
        gt_name = os.path.join(self.cf.dataset_path, self.folder, self.indices[idx], 'GT.npy')
        
        image_np = np.load(img_name)       
        image2_np = np.load(img_name2)       
        image1_np = np.load(img_name1)   
        
#        image2_np=np.zeros_like(image1_np)
        image1_np=np.zeros_like(image_np)
        
        gt_np = np.load(gt_name)>0 #check this        
        if self.cf.da_resize:
            image_np = resize(image_np,(self.cf.size_train[0],self.cf.size_train[1],self.cf.size_train[2]),mode='constant')
            image2_np = resize(image2_np,(self.cf.size_train[0],self.cf.size_train[1],self.cf.size_train[2]),mode='constant')
            image1_np = resize(image1_np,(self.cf.size_train[0],self.cf.size_train[1],self.cf.size_train[2]),mode='constant')            
            gt_np = resize(gt_np,(self.cf.size_train[0],self.cf.size_train[1],self.cf.size_train[2]),order=0,mode='constant')       
        
        if self.cf.da_norm:
             image_np = normalize(image_np)
             image2_np = normalize(image2_np)
#             image1_np = normalize(image1_np)
        
                      
        #transforming them if da (custom transforms)
        if self.transform and self.phase == 'train': 
            image_np, image2_np, image1_np, gt_np = self.__transforms(image_np, image2_np, image1_np, gt_np)

           
        image = torch.from_numpy(image_np.copy()).unsqueeze(0).float()
        image2 = torch.from_numpy(image2_np.copy()).unsqueeze(0).float()
        image1 = torch.from_numpy(image1_np.copy()).unsqueeze(0).float()

        gt = torch.from_numpy(gt_np.copy().astype(float)).unsqueeze(0).float()
        images=torch.cat((image,image2,image1),0)
        data = (images,gt)
        patient_name = self.indices[idx]
        return data, patient_name
        
    def __transforms(self, im, im2, im1, gt):
        
        ##Image level
        if self.cf.da_ver_flip and np.random.random() < 0.5:
            im, im2, im1, gt = vertical_flip3(im, im2, im1, gt)  
        if self.cf.da_rotate:    
            im, im2, im1, gt = random_rotation3(im, im2, im1, gt, self.cf.da_rotate)        
        if self.cf.da_deform and np.random.random() < 0.5:
             im, im2, im1, gt = elasticdeform.deform_random_grid([im, im2, im1, gt], sigma=self.cf.da_deform, points=3, order=[3,3,3,0])
        return im, im2, im1, gt