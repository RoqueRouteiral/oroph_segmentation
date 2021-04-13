import os
import glob
import torch
import random
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tools.misc import get_prob_map, get_thick_edge
#from skimage import exposure
#from skimage.transform import resize
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
        img_name = os.path.join(self.cf.dataset_path, self.folder, self.indices[idx], 'T1c.npy')
        img2_name = os.path.join(self.cf.dataset_path, self.folder, self.indices[idx], 'T2.npy')
        img1_name = os.path.join(self.cf.dataset_path, self.folder, self.indices[idx], 'T1.npy')
        gt_name = os.path.join(self.cf.dataset_path, self.folder, self.indices[idx], 'GT.npy')
        
        image_np = np.load(img_name)        
        image2_np = np.load(img2_name)
        image1_np = np.load(img1_name)      
        gt_np = np.load(gt_name)>0 #check this  

#        #transforming them if da (custom transforms)
#        if self.transform and self.phase == 'train': 
#            image_np, image2_np, image1_np, gt_np = self.__transforms(image_np, image2_np, image1_np, gt_np)

        gt_dist = ndimage.distance_transform_edt(gt_np==0)
        gt_dist = gt_dist<1
        coor = np.where(gt_dist)
        if self.phase == 'train':
#            borders=np.random.randint(0, high=self.cf.max_shift, size=6)
            borders= 8 * np.random.randn(6) + 17
        else:
            borders=np.ones(6).astype(int)*8

            
        new_borders=np.array([np.min(coor[0])-borders[0],np.max(coor[0])+borders[1]+1,
                              np.min(coor[1])-borders[2],np.max(coor[1])+borders[3]+1,
                              np.min(coor[2])-borders[4],np.max(coor[2])+borders[5]+1])
        #print(new_borders)
        new_borders[new_borders<0]=0
        new_borders[1]=np.min((image_np.shape[0],new_borders[1]))
        new_borders[3]=np.min((image_np.shape[1],new_borders[3]))
        new_borders[5]=np.min((image_np.shape[2],new_borders[5]))        
            
        #Now we make sure that they are not more than 112 of width
        if new_borders[1]-new_borders[0]>112:new_borders[1]=new_borders[1]-(new_borders[1]-new_borders[0]-112)
        if new_borders[3]-new_borders[2]>112:new_borders[3]=new_borders[3]-(new_borders[3]-new_borders[2]-112)
        if new_borders[5]-new_borders[4]>112:new_borders[5]=new_borders[5]-(new_borders[5]-new_borders[4]-112)
        #print(new_borders)
        new_borders=new_borders.astype(int)
        cropped_img = image_np[new_borders[0]:new_borders[1], new_borders[2]:new_borders[3], new_borders[4]:new_borders[5]]
        cropped2_img = image2_np[new_borders[0]:new_borders[1], new_borders[2]:new_borders[3], new_borders[4]:new_borders[5]]
        cropped1_img = image1_np[new_borders[0]:new_borders[1], new_borders[2]:new_borders[3], new_borders[4]:new_borders[5]]
        
        cropped_gt = gt_np[new_borders[0]:new_borders[1], new_borders[2]:new_borders[3], new_borders[4]:new_borders[5]]

        cropped_img=np.pad(cropped_img,((0,112-(new_borders[1]-new_borders[0])),(0,112-(new_borders[3]-new_borders[2])),(0,112-(new_borders[5]-new_borders[4]))),mode='constant')
        cropped2_img=np.pad(cropped2_img,((0,112-(new_borders[1]-new_borders[0])),(0,112-(new_borders[3]-new_borders[2])),(0,112-(new_borders[5]-new_borders[4]))),mode='constant')
        cropped1_img=np.pad(cropped1_img,((0,112-(new_borders[1]-new_borders[0])),(0,112-(new_borders[3]-new_borders[2])),(0,112-(new_borders[5]-new_borders[4]))),mode='constant')

        cropped_gt=np.pad(cropped_gt,((0,112-(new_borders[1]-new_borders[0])),(0,112-(new_borders[3]-new_borders[2])),(0,112-(new_borders[5]-new_borders[4]))),mode='constant')        

        image_np=cropped_img
        image2_np=cropped2_img
        image1_np=cropped1_img
        
        gt_np=cropped_gt
#        if self.cf.da_resize:
#            image_np = resize(image_np,(self.cf.size_train[0],self.cf.size_train[1],self.cf.size_train[2]),mode='constant')
#            gt_np = resize(gt_np,(self.cf.size_train[0],self.cf.size_train[1],self.cf.size_train[2]),mode='constant')        
        
        if self.cf.da_norm:
             image_np = normalize(image_np)
             image2_np = normalize(image2_np)
             image1_np = normalize(image1_np)

#        image_np = image_np.transpose(2, 0, 1)
#        gt_np = gt_np.transpose(2, 0, 1)
#                      
#        #transforming them if da (custom transforms)
#        if self.transform and self.phase == 'train': 
#            image_np, gt_np = self.__transforms(image_np, gt_np)

        
        image = torch.from_numpy(image_np.copy()).unsqueeze(0).float()
        image2 = torch.from_numpy(image2_np.copy()).unsqueeze(0).float()
        image1 = torch.from_numpy(image1_np.copy()).unsqueeze(0).float()
        
        images=torch.cat((image,image2,image1),0)
        gt = torch.from_numpy(gt_np.copy().astype(float)).unsqueeze(0).float()
    
        data = (images,gt)
        patient_name = self.indices[idx]

        return data, patient_name
        
    def __transforms(self, im, im2, im1, gt):
        
        ##Image level
        if self.cf.da_ver_flip and np.random.random() < 0.5:
            im, im2, im1, gt = vertical_flip3(im, im2, im1, gt)  
        if self.cf.da_rotate:    
            im, im2, im1, gt = random_rotation3(im, im2, im1, gt, self.cf.da_rotate)      
        if self.cf.da_deform:
             im, im2, im1, gt = elasticdeform.deform_random_grid([im, im2, im1, gt], sigma=self.cf.da_deform, points=3, order=[3, 3 ,3 , 0])
         
        return im, im2, im1, gt