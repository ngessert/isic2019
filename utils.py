import os
import torch
#import pandas as pd
from skimage import io, transform
import scipy
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, utils
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import math
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import types
from auto_augment import AutoAugment, Cutout

# Define ISIC Dataset Class
class ISICDataset(Dataset):
    """ISIC dataset."""

    def __init__(self, mdlParams, indSet):
        """
        Args:
            mdlParams (dict): Configuration for loading
            indSet (string): Indicates train, val, test
        """
        # Mdlparams
        self.mdlParams = mdlParams
        # Number of classes
        self.numClasses = mdlParams['numClasses']
        # Model input size
        self.input_size = (np.int32(mdlParams['input_size'][0]),np.int32(mdlParams['input_size'][1]))      
        # Whether or not to use ordered cropping 
        self.orderedCrop = mdlParams['orderedCrop']   
        # Number of crops for multi crop eval
        self.multiCropEval = mdlParams['multiCropEval']   
        # Whether during training same-sized crops should be used
        self.same_sized_crop = mdlParams['same_sized_crops']    
        # Only downsample
        self.only_downsmaple = mdlParams.get('only_downsmaple',False)   
        # Potential class balancing option 
        self.balancing = mdlParams['balance_classes']
        # Whether data should be preloaded
        self.preload = mdlParams['preload']
        # Potentially subtract a mean
        self.subtract_set_mean = mdlParams['subtract_set_mean']
        # Potential switch for evaluation on the training set
        self.train_eval_state = mdlParams['trainSetState']   
        # Potential setMean to deduce from channels
        self.setMean = mdlParams['setMean'].astype(np.float32)
        # Current indSet = 'trainInd'/'valInd'/'testInd'
        self.indices = mdlParams[indSet]  
        self.indSet = indSet
        # feature scaling for meta
        if mdlParams.get('meta_features',None) is not None and mdlParams['scale_features']:
            self.feature_scaler = mdlParams['feature_scaler_meta']
        if self.balancing == 3 and indSet == 'trainInd':
            # Sample classes equally for each batch
            # First, split set by classes
            not_one_hot = np.argmax(mdlParams['labels_array'],1)
            self.class_indices = []
            for i in range(mdlParams['numClasses']):
                self.class_indices.append(np.where(not_one_hot==i)[0])
                # Kick out non-trainind indices
                self.class_indices[i] = np.setdiff1d(self.class_indices[i],mdlParams['valInd'])
                # And test indices
                if 'testInd' in mdlParams:
                    self.class_indices[i] = np.setdiff1d(self.class_indices[i],mdlParams['testInd'])
            # Now sample indices equally for each batch by repeating all of them to have the same amount as the max number
            indices = []
            max_num = np.max([len(x) for x in self.class_indices])
            # Go thourgh all classes
            for i in range(mdlParams['numClasses']):
                count = 0
                class_count = 0
                max_num_curr_class = len(self.class_indices[i])
                # Add examples until we reach the maximum
                while(count < max_num):
                    # Start at the beginning, if we are through all available examples
                    if class_count == max_num_curr_class:
                        class_count = 0
                    indices.append(self.class_indices[i][class_count])
                    count += 1
                    class_count += 1
            print("Largest class",max_num,"Indices len",len(indices))
            print("Intersect val",np.intersect1d(indices,mdlParams['valInd']),"Intersect Testind",np.intersect1d(indices,mdlParams['testInd']))
            # Set labels/inputs
            self.labels = mdlParams['labels_array'][indices,:]
            self.im_paths = np.array(mdlParams['im_paths'])[indices].tolist()     
            # Normal train proc
            if self.same_sized_crop:
                cropping = transforms.RandomCrop(self.input_size)
            elif self.only_downsmaple:
                cropping = transforms.Resize(self.input_size)
            else:
                cropping = transforms.RandomResizedCrop(self.input_size[0])
            # All transforms
            self.composed = transforms.Compose([
                    cropping,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=32. / 255.,saturation=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
                    ])                                
        elif self.orderedCrop and (indSet == 'valInd' or self.train_eval_state  == 'eval' or indSet == 'testInd'):
            # Also flip on top            
            if mdlParams.get('eval_flipping',0) > 1:
                # Complete labels array, only for current indSet, repeat for multiordercrop
                inds_rep = np.repeat(mdlParams[indSet], mdlParams['multiCropEval']*mdlParams['eval_flipping'])
                self.labels = mdlParams['labels_array'][inds_rep,:]
                # meta
                if mdlParams.get('meta_features',None) is not None:
                    self.meta_data = mdlParams['meta_array'][inds_rep,:]    
                # Path to images for loading, only for current indSet, repeat for multiordercrop
                self.im_paths = np.array(mdlParams['im_paths'])[inds_rep].tolist()
                print("len im path",len(self.im_paths))                
                if self.mdlParams.get('var_im_size',False):
                    self.cropPositions = np.tile(mdlParams['cropPositions'][mdlParams[indSet],:,:],(1,mdlParams['eval_flipping'],1))
                    self.cropPositions = np.reshape(self.cropPositions,[mdlParams['multiCropEval']*mdlParams['eval_flipping']*mdlParams[indSet].shape[0],2])
                    #self.cropPositions = np.repeat(self.cropPositions, (mdlParams['eval_flipping'],1))
                    #print("CP examples",self.cropPositions[:50,:])
                else:
                    self.cropPositions = np.tile(mdlParams['cropPositions'], (mdlParams['eval_flipping']*mdlParams[indSet].shape[0],1))
                # Flip states
                if mdlParams['eval_flipping'] == 2:
                    self.flipPositions = np.array([0,1])
                elif mdlParams['eval_flipping'] == 3:
                    self.flipPositions = np.array([0,1,2])
                elif mdlParams['eval_flipping'] == 4:
                    self.flipPositions = np.array([0,1,2,3])                    
                self.flipPositions = np.repeat(self.flipPositions, mdlParams['multiCropEval'])
                self.flipPositions = np.tile(self.flipPositions, mdlParams[indSet].shape[0])
                print("Crop positions shape",self.cropPositions.shape,"flip pos shape",self.flipPositions.shape)
                print("Flip example",self.flipPositions[:30])
            else:
                # Complete labels array, only for current indSet, repeat for multiordercrop
                inds_rep = np.repeat(mdlParams[indSet], mdlParams['multiCropEval'])
                self.labels = mdlParams['labels_array'][inds_rep,:]
                # meta
                if mdlParams.get('meta_features',None) is not None:
                    self.meta_data = mdlParams['meta_array'][inds_rep,:]                 
                # Path to images for loading, only for current indSet, repeat for multiordercrop
                self.im_paths = np.array(mdlParams['im_paths'])[inds_rep].tolist()
                print("len im path",len(self.im_paths))
                # Set up crop positions for every sample                
                if self.mdlParams.get('var_im_size',False):
                    self.cropPositions = np.reshape(mdlParams['cropPositions'][mdlParams[indSet],:,:],[mdlParams['multiCropEval']*mdlParams[indSet].shape[0],2])
                    #print("CP examples",self.cropPositions[:50,:])
                else:
                    self.cropPositions = np.tile(mdlParams['cropPositions'], (mdlParams[indSet].shape[0],1))
                print("CP",self.cropPositions.shape)
            #print("CP Example",self.cropPositions[0:len(mdlParams['cropPositions']),:])          
            # Set up transforms
            self.norm = transforms.Normalize(np.float32(self.mdlParams['setMean']),np.float32(self.mdlParams['setStd']))
            self.trans = transforms.ToTensor()
        elif indSet == 'valInd' or indSet == 'testInd':
            if self.multiCropEval == 0:
                if self.only_downsmaple:
                    self.cropping = transforms.Resize(self.input_size)
                else:
                    self.cropping = transforms.Compose([transforms.CenterCrop(np.int32(self.input_size[0]*1.5)),transforms.Resize(self.input_size)])
                # Complete labels array, only for current indSet
                self.labels = mdlParams['labels_array'][mdlParams[indSet],:]
                # meta
                if mdlParams.get('meta_features',None) is not None:
                    self.meta_data = mdlParams['meta_array'][mdlParams[indSet],:]                 
                # Path to images for loading, only for current indSet
                self.im_paths = np.array(mdlParams['im_paths'])[mdlParams[indSet]].tolist()                   
            else:
                # Deterministic processing
                if self.mdlParams.get('deterministic_eval',False):
                    total_len_per_im = mdlParams['numCropPositions']*len(mdlParams['cropScales'])*mdlParams['cropFlipping']                    
                    # Actual transforms are functionally applied at forward pass
                    self.cropPositions = np.zeros([total_len_per_im,3])
                    ind = 0
                    for i in range(mdlParams['numCropPositions']):
                        for j in range(len(mdlParams['cropScales'])):
                            for k in range(mdlParams['cropFlipping']):
                                self.cropPositions[ind,0] = i
                                self.cropPositions[ind,1] = mdlParams['cropScales'][j]
                                self.cropPositions[ind,2] = k
                                ind += 1
                    # Complete labels array, only for current indSet, repeat for multiordercrop
                    print("crops per image",total_len_per_im)
                    self.cropPositions = np.tile(self.cropPositions, (mdlParams[indSet].shape[0],1))
                    inds_rep = np.repeat(mdlParams[indSet], total_len_per_im)
                    self.labels = mdlParams['labels_array'][inds_rep,:]
                    # meta
                    if mdlParams.get('meta_features',None) is not None:
                        self.meta_data = mdlParams['meta_array'][inds_rep,:]                     
                    # Path to images for loading, only for current indSet, repeat for multiordercrop
                    self.im_paths = np.array(mdlParams['im_paths'])[inds_rep].tolist()
                else:
                    self.cropping = transforms.RandomResizedCrop(self.input_size[0],scale=(mdlParams.get('scale_min',0.08),1.0))
                    # Complete labels array, only for current indSet, repeat for multiordercrop
                    inds_rep = np.repeat(mdlParams[indSet], mdlParams['multiCropEval'])
                    self.labels = mdlParams['labels_array'][inds_rep,:]
                    # meta
                    if mdlParams.get('meta_features',None) is not None:
                        self.meta_data = mdlParams['meta_array'][inds_rep,:]                    
                    # Path to images for loading, only for current indSet, repeat for multiordercrop
                    self.im_paths = np.array(mdlParams['im_paths'])[inds_rep].tolist()
            print(len(self.im_paths))  
            # Set up transforms
            self.norm = transforms.Normalize(np.float32(self.mdlParams['setMean']),np.float32(self.mdlParams['setStd']))
            self.trans = transforms.ToTensor()                   
        else:
            all_transforms = []
            # Normal train proc
            if self.same_sized_crop:
                all_transforms.append(transforms.RandomCrop(self.input_size))
            elif self.only_downsmaple:
                all_transforms.append(transforms.Resize(self.input_size))
            else:
                all_transforms.append(transforms.RandomResizedCrop(self.input_size[0],scale=(mdlParams.get('scale_min',0.08),1.0)))
            if mdlParams.get('flip_lr_ud',False):
                all_transforms.append(transforms.RandomHorizontalFlip())
                all_transforms.append(transforms.RandomVerticalFlip())
            # Full rot
            if mdlParams.get('full_rot',0) > 0:
                if mdlParams.get('scale',False):
                    all_transforms.append(transforms.RandomChoice([transforms.RandomAffine(mdlParams['full_rot'], scale=mdlParams['scale'], shear=mdlParams.get('shear',0), resample=Image.NEAREST),
                                                                transforms.RandomAffine(mdlParams['full_rot'],scale=mdlParams['scale'],shear=mdlParams.get('shear',0), resample=Image.BICUBIC),
                                                                transforms.RandomAffine(mdlParams['full_rot'],scale=mdlParams['scale'],shear=mdlParams.get('shear',0), resample=Image.BILINEAR)])) 
                else:
                    all_transforms.append(transforms.RandomChoice([transforms.RandomRotation(mdlParams['full_rot'], resample=Image.NEAREST),
                                                                transforms.RandomRotation(mdlParams['full_rot'], resample=Image.BICUBIC),
                                                                transforms.RandomRotation(mdlParams['full_rot'], resample=Image.BILINEAR)]))    
            # Color distortion
            if mdlParams.get('full_color_distort') is not None:
                all_transforms.append(transforms.ColorJitter(brightness=mdlParams.get('brightness_aug',32. / 255.),saturation=mdlParams.get('saturation_aug',0.5), contrast = mdlParams.get('contrast_aug',0.5), hue = mdlParams.get('hue_aug',0.2)))
            else:
                all_transforms.append(transforms.ColorJitter(brightness=32. / 255.,saturation=0.5))   
            # Autoaugment
            if self.mdlParams.get('autoaugment',False):
                all_transforms.append(AutoAugment())             
            # Cutout
            if self.mdlParams.get('cutout',0) > 0:
                all_transforms.append(Cutout_v0(n_holes=1,length=self.mdlParams['cutout']))                             
            # Normalize
            all_transforms.append(transforms.ToTensor())
            all_transforms.append(transforms.Normalize(np.float32(self.mdlParams['setMean']),np.float32(self.mdlParams['setStd'])))            
            # All transforms
            self.composed = transforms.Compose(all_transforms)                  
            # Complete labels array, only for current indSet
            self.labels = mdlParams['labels_array'][mdlParams[indSet],:]
            # meta
            if mdlParams.get('meta_features',None) is not None:
                self.meta_data = mdlParams['meta_array'][mdlParams[indSet],:]            
            # Path to images for loading, only for current indSet
            self.im_paths = np.array(mdlParams['im_paths'])[mdlParams[indSet]].tolist()
        # Potentially preload
        if self.preload:
            self.im_list = []
            for i in range(len(self.im_paths)):
                self.im_list.append(Image.open(self.im_paths[i]))
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # Load image
        if self.preload:
            x = self.im_list[idx]
        else:
            x = Image.open(self.im_paths[idx])
            if self.mdlParams.get('resize_large_ones',0) > 0 and (x.size[0] == self.mdlParams['large_size'] and x.size[1] == self.mdlParams['large_size']):
                width = self.mdlParams['resize_large_ones']
                height = self.mdlParams['resize_large_ones']
                #height = (self.mdlParams['resize_large_ones']/self.mdlParams['large_size'])*x.size[1]
                x = x.resize((width,height),Image.BILINEAR)
            if self.mdlParams['input_size'][0] >= 224 and self.mdlParams['orderedCrop']:
                if x.size[0] < self.mdlParams['input_size'][0]:
                    new_height = int(self.mdlParams['input_size'][0]/float(x.size[0]))*x.size[1]
                    new_width = self.mdlParams['input_size'][0]
                    x = x.resize((new_width,new_height),Image.BILINEAR)
                if x.size[1] < self.mdlParams['input_size'][0]:
                    new_width = int(self.mdlParams['input_size'][0]/float(x.size[1]))*x.size[0]
                    new_height = self.mdlParams['input_size'][0]
                    x = x.resize((new_width,new_height),Image.BILINEAR)               
        # Get label
        y = self.labels[idx,:]
        # meta
        if self.mdlParams.get('meta_features',None) is not None:
            x_meta = self.meta_data[idx,:].copy()         
        # Transform data based on whether train or not train. If train, also check if its train train or train inference
        if self.orderedCrop and (self.indSet == 'valInd' or self.indSet == 'testInd' or self.train_eval_state == 'eval'):
            # Apply ordered cropping to validation or test set
            # Get current crop position
            x_loc = self.cropPositions[idx,0]
            y_loc = self.cropPositions[idx,1]
            # scale
            if self.mdlParams.get('meta_features',None) is not None and self.mdlParams['scale_features']:
                x_meta = np.squeeze(self.feature_scaler.transform(np.expand_dims(x_meta,0)))            
            if self.mdlParams.get('trans_norm_first',False):
                # First, to pytorch tensor (0.0-1.0)
                x = self.trans(x)
                # Normalize
                x = self.norm(x)   
                #print(self.im_paths[idx])
                #print("Before",x.size(),"xloc",x_loc,"y_loc",y_loc)
                if self.mdlParams.get('eval_flipping',0) > 1:
                    if self.flipPositions[idx] == 1:
                        x = torch.flip(x,(1,))
                    elif self.flipPositions[idx] == 2:
                        x = torch.flip(x,(2,))
                    elif self.flipPositions[idx] == 3:
                        x = torch.flip(x,(1,2))
                #print((x_loc-np.int32(self.input_size[0]/2.)),(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],(y_loc-np.int32(self.input_size[1]/2.)),(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1])                
                x = x[:,np.int32(x_loc-(self.input_size[0]/2.)):np.int32(x_loc-(self.input_size[0]/2.))+self.input_size[0],
                        np.int32(y_loc-(self.input_size[1]/2.)):np.int32(y_loc-(self.input_size[1]/2.))+self.input_size[1]]                 
                #print("After",x.size())           
            else:
                # Then, apply current crop
                #print("Before",x.size(),"xloc",x_loc,"y_loc",y_loc)
                #print((x_loc-np.int32(self.input_size[0]/2.)),(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],(y_loc-np.int32(self.input_size[1]/2.)),(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1])
                x = Image.fromarray(np.array(x)[(x_loc-np.int32(self.input_size[0]/2.)):(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],
                        (y_loc-np.int32(self.input_size[1]/2.)):(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1],:])
                # First, to pytorch tensor (0.0-1.0)
                x = self.trans(x)
                # Normalize
                x = self.norm(x)            
            #print("After",x.size())
        elif self.indSet == 'valInd' or self.indSet == 'testInd':
            if self.mdlParams.get('deterministic_eval',False):
                crop = self.cropPositions[idx,0]   
                scale = self.cropPositions[idx,1]
                flipping = self.cropPositions[idx,2]
                if flipping == 1:
                    # Left flip
                    x = transforms.functional.hflip(x)
                elif flipping == 2:
                    # Right flip
                    x = transforms.functional.vflip(x)
                elif flipping == 3:
                    # Both flip
                    x = transforms.functional.hflip(x)
                    x = transforms.functional.vflip(x)                    
                # Scale
                if int(scale*x.size[0]) > self.input_size[0] and int(scale*x.size[1]) > self.input_size[1]:
                    x = transforms.functional.resize(x,(int(scale*x.size[0]),int(scale*x.size[1])))
                else:
                    x = transforms.functional.resize(x,(self.input_size[0],self.input_size[1]))
                # Crop
                if crop == 0:
                    # Center
                    x = transforms.functional.center_crop(x,self.input_size[0])
                elif crop == 1:
                    # upper left
                    x = transforms.functional.crop(x, self.mdlParams['offset_crop']*x.size[0], self.mdlParams['offset_crop']*x.size[1], self.input_size[0],self.input_size[1])
                elif crop == 2:
                    # lower left
                    x = transforms.functional.crop(x, self.mdlParams['offset_crop']*x.size[0], (1.0-self.mdlParams['offset_crop'])*x.size[1]-self.input_size[1], self.input_size[0],self.input_size[1]) 
                elif crop == 3:
                    # upper right
                    x = transforms.functional.crop(x, (1.0-self.mdlParams['offset_crop'])*x.size[0]-self.input_size[0], self.mdlParams['offset_crop']*x.size[1], self.input_size[0],self.input_size[1])  
                elif crop == 4:
                    # lower right
                    x = transforms.functional.crop(x, (1.0-self.mdlParams['offset_crop'])*x.size[0]-self.input_size[0], (1.0-self.mdlParams['offset_crop'])*x.size[1]-self.input_size[1], self.input_size[0],self.input_size[1])       
            else:
                x = self.cropping(x)        
            # To pytorch tensor (0.0-1.0)
            x = self.trans(x)
            x = self.norm(x)    
            # scale
            if self.mdlParams.get('meta_features',None) is not None and self.mdlParams['scale_features']:
                x_meta = np.squeeze(self.feature_scaler.transform(np.expand_dims(x_meta,0)))                          
        else:
            # Apply
            x = self.composed(x)
            # meta augment
            if self.mdlParams.get('meta_features',None) is not None:
                if self.mdlParams['drop_augment'] > 0:
                    # randomly deactivate a feature
                    # age
                    if torch.rand(1) < self.mdlParams['drop_augment']:
                        if 'age_oh' in self.mdlParams['meta_features']:
                            x_meta[0:self.mdlParams['meta_feature_sizes'][0]] = np.zeros([self.mdlParams['meta_feature_sizes'][0]])
                        else:
                            x_meta[0] = -5
                    if torch.rand(1) < self.mdlParams['drop_augment']:
                        if 'loc_oh' in self.mdlParams['meta_features']:   
                            x_meta[self.mdlParams['meta_feature_sizes'][0]:self.mdlParams['meta_feature_sizes'][0]+self.mdlParams['meta_feature_sizes'][1]] = np.zeros([self.mdlParams['meta_feature_sizes'][1]])
                    if torch.rand(1) < self.mdlParams['drop_augment']:
                        if 'sex_oh' in self.mdlParams['meta_features']:   
                            x_meta[self.mdlParams['meta_feature_sizes'][0]+self.mdlParams['meta_feature_sizes'][1]:self.mdlParams['meta_feature_sizes'][0]+self.mdlParams['meta_feature_sizes'][1]+self.mdlParams['meta_feature_sizes'][2]] = np.zeros([self.mdlParams['meta_feature_sizes'][2]]) 
                # scale
                if self.mdlParams['scale_features']:
                    x_meta = np.squeeze(self.feature_scaler.transform(np.expand_dims(x_meta,0)))                         
        # Transform y
        y = np.argmax(y)
        y = np.int64(y)
        if self.mdlParams.get('meta_features',None) is not None:
            x_meta = np.float32(x_meta) 
        if self.mdlParams.get('eval_flipping',0) > 1:
            return x, y, idx, self.flipPositions[idx]
        else:
            if self.mdlParams.get('meta_features',None) is not None:
                return (x, x_meta), y, idx
            else:
                return x, y, idx




class Cutout_v0(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        img = np.array(img)
        #print(img.shape)
        h = img.shape[0]
        w = img.shape[1]

        mask = np.ones((h, w), np.uint8)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        #mask = torch.from_numpy(mask)
        #mask = mask.expand_as(img)
        img = img * np.expand_dims(mask,axis=2)
        img = Image.fromarray(img)
        return img    

# Sampler for balanced sampling
class StratifiedSampler(torch.utils.data.sampler.Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, mdlParams):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.dataset_len = len(mdlParams['trainInd'])
        self.numClasses = mdlParams['numClasses']
        self.trainInd = mdlParams['trainInd']
        # Sample classes equally for each batch
        # First, split set by classes
        not_one_hot = np.argmax(mdlParams['labels_array'][mdlParams['trainInd'],:],1)
        self.class_indices = []
        for i in range(mdlParams['numClasses']):
            self.class_indices.append(np.where(not_one_hot==i)[0])
        self.current_class_ind = 0
        self.current_in_class_ind = np.zeros([mdlParams['numClasses']],dtype=int)

    def gen_sample_array(self):
        # Shuffle all classes first
        for i in range(self.numClasses):
            np.random.shuffle(self.class_indices[i])
        # Construct indset
        indices = np.zeros([self.dataset_len],dtype=np.int32)
        ind = 0
        while(ind < self.dataset_len):
            indices[ind] = self.class_indices[self.current_class_ind][self.current_in_class_ind[self.current_class_ind]]
            # Take care of in-class index
            if self.current_in_class_ind[self.current_class_ind] == len(self.class_indices[self.current_class_ind])-1:
                self.current_in_class_ind[self.current_class_ind] = 0
                # Shuffle
                np.random.shuffle(self.class_indices[self.current_class_ind])
            else:
                self.current_in_class_ind[self.current_class_ind] += 1
            # Take care of overall class ind
            if self.current_class_ind == self.numClasses-1:
                self.current_class_ind = 0
            else:
                self.current_class_ind += 1
            ind += 1
        return indices

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return self.dataset_len 

class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        #print("before gather",logpt)
        #print("target",target)
        logpt = logpt.gather(1,target)
        #print("after gather",logpt)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            #print("alpha",self.alpha)
            #print("gathered",at)
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def getErrClassification_mgpu(mdlParams, indices, modelVars, exclude_class=None):
    """Helper function to return the error of a set
    Args:
      mdlParams: dictionary, configuration file
      indices: string, either "trainInd", "valInd" or "testInd"
    Returns:
      loss: float, avg loss
      acc: float, accuracy
      sensitivity: float, sensitivity
      spec: float, specificity
      conf: float matrix, confusion matrix
    """
    # Set up sizes
    if indices == 'trainInd':
        numBatches = int(math.floor(len(mdlParams[indices])/mdlParams['batchSize']/len(mdlParams['numGPUs'])))
    else:
        numBatches = int(math.ceil(len(mdlParams[indices])/mdlParams['batchSize']/len(mdlParams['numGPUs'])))
    # Consider multi-crop case
    if mdlParams.get('eval_flipping',0) > 1 and mdlParams.get('multiCropEval',0) > 0:
        loss_all = np.zeros([numBatches])
        predictions = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
        targets = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])        
        loss_mc = np.zeros([len(mdlParams[indices])*mdlParams['eval_flipping']])
        predictions_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval'],mdlParams['eval_flipping']])
        targets_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval'],mdlParams['eval_flipping']])  
        # Very suboptimal method
        ind = -1
        for i, (inputs, labels, inds, flip_ind) in enumerate(modelVars['dataloader_'+indices]):
            if flip_ind[0] != np.mean(np.array(flip_ind)):
                print("Problem with flipping",flip_ind)
            if flip_ind[0] == 0:
                ind += 1
            # Get data
            if mdlParams.get('meta_features',None) is not None: 
                inputs[0] = inputs[0].cuda()
                inputs[1] = inputs[1].cuda()
            else:            
                inputs = inputs.to(modelVars['device'])
            labels = labels.to(modelVars['device'])       
            # Not sure if thats necessary
            modelVars['optimizer'].zero_grad()    
            with torch.set_grad_enabled(False):
                # Get outputs
                if mdlParams.get('aux_classifier',False):
                    outputs, outputs_aux = modelVars['model'](inputs)
                    if mdlParams['eval_aux_classifier']:
                        outputs = outputs_aux
                else:
                    outputs = modelVars['model'](inputs)
                preds = modelVars['softmax'](outputs)      
                # Loss
                loss = modelVars['criterion'](outputs, labels)           
            # Write into proper arrays
            loss_mc[ind] = np.mean(loss.cpu().numpy())
            predictions_mc[ind,:,:,flip_ind[0]] = np.transpose(preds.cpu().numpy())
            tar_not_one_hot = labels.data.cpu().numpy()
            tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
            tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
            targets_mc[ind,:,:,flip_ind[0]] = np.transpose(tar)
        # Targets stay the same
        targets = targets_mc[:,:,0,0]
        # reshape preds
        predictions_mc = np.reshape(predictions_mc,[predictions_mc.shape[0],predictions_mc.shape[1],mdlParams['multiCropEval']*mdlParams['eval_flipping']])
        if mdlParams['voting_scheme'] == 'vote':
            # Vote for correct prediction
            print("Pred Shape",predictions_mc.shape)
            predictions_mc = np.argmax(predictions_mc,1)    
            print("Pred Shape",predictions_mc.shape) 
            for j in range(predictions_mc.shape[0]):
                predictions[j,:] = np.bincount(predictions_mc[j,:],minlength=mdlParams['numClasses'])   
            print("Pred Shape",predictions.shape) 
        elif mdlParams['voting_scheme'] == 'average':
            predictions = np.mean(predictions_mc,2)        
    elif mdlParams.get('multiCropEval',0) > 0:
        loss_all = np.zeros([numBatches])
        predictions = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
        targets = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])        
        loss_mc = np.zeros([len(mdlParams[indices])])
        predictions_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval']])
        targets_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval']])   
        for i, (inputs, labels, inds) in enumerate(modelVars['dataloader_'+indices]):
            # Get data
            if mdlParams.get('meta_features',None) is not None: 
                inputs[0] = inputs[0].cuda()
                inputs[1] = inputs[1].cuda()
            else:            
                inputs = inputs.to(modelVars['device'])
            labels = labels.to(modelVars['device'])       
            # Not sure if thats necessary
            modelVars['optimizer'].zero_grad()    
            with torch.set_grad_enabled(False):
                # Get outputs
                if mdlParams.get('aux_classifier',False):
                    outputs, outputs_aux = modelVars['model'](inputs)
                    if mdlParams['eval_aux_classifier']:
                        outputs = outputs_aux
                else:
                    outputs = modelVars['model'](inputs)
                preds = modelVars['softmax'](outputs)      
                # Loss
                loss = modelVars['criterion'](outputs, labels)           
            # Write into proper arrays
            loss_mc[i] = np.mean(loss.cpu().numpy())
            predictions_mc[i,:,:] = np.transpose(preds.cpu().numpy())
            tar_not_one_hot = labels.data.cpu().numpy()
            tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
            tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
            targets_mc[i,:,:] = np.transpose(tar)
        # Targets stay the same
        targets = targets_mc[:,:,0]
        if mdlParams['voting_scheme'] == 'vote':
            # Vote for correct prediction
            print("Pred Shape",predictions_mc.shape)
            predictions_mc = np.argmax(predictions_mc,1)    
            print("Pred Shape",predictions_mc.shape) 
            for j in range(predictions_mc.shape[0]):
                predictions[j,:] = np.bincount(predictions_mc[j,:],minlength=mdlParams['numClasses'])   
            print("Pred Shape",predictions.shape) 
        elif mdlParams['voting_scheme'] == 'average':
            predictions = np.mean(predictions_mc,2)
    else:    
        if mdlParams.get('model_type_cnn') is not None and mdlParams['numRandValSeq'] > 0:
            loss_all = np.zeros([numBatches])
            predictions = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
            targets = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])        
            loss_mc = np.zeros([len(mdlParams[indices])])
            predictions_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['numRandValSeq']])
            targets_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['numRandValSeq']])   
            for i, (inputs, labels, inds) in enumerate(modelVars['dataloader_'+indices]):
                # Get data
                if mdlParams.get('meta_features',None) is not None: 
                    inputs[0] = inputs[0].cuda()
                    inputs[1] = inputs[1].cuda()
                else:            
                    inputs = inputs.to(modelVars['device'])
                labels = labels.to(modelVars['device'])       
                # Not sure if thats necessary
                modelVars['optimizer'].zero_grad()    
                with torch.set_grad_enabled(False):
                    # Get outputs
                    if mdlParams.get('aux_classifier',False):
                        outputs, outputs_aux = modelVars['model'](inputs)
                        if mdlParams['eval_aux_classifier']:
                            outputs = outputs_aux
                    else:
                        outputs = modelVars['model'](inputs)
                    preds = modelVars['softmax'](outputs)      
                    # Loss
                    loss = modelVars['criterion'](outputs, labels)           
                # Write into proper arrays
                loss_mc[i] = np.mean(loss.cpu().numpy())
                predictions_mc[i,:,:] = np.transpose(preds)
                tar_not_one_hot = labels.data.cpu().numpy()
                tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
                targets_mc[i,:,:] = np.transpose(tar)
            # Targets stay the same
            targets = targets_mc[:,:,0]
            if mdlParams['voting_scheme'] == 'vote':
                # Vote for correct prediction
                print("Pred Shape",predictions_mc.shape)
                predictions_mc = np.argmax(predictions_mc,1)    
                print("Pred Shape",predictions_mc.shape) 
                for j in range(predictions_mc.shape[0]):
                    predictions[j,:] = np.bincount(predictions_mc[j,:],minlength=mdlParams['numClasses'])   
                print("Pred Shape",predictions.shape) 
            elif mdlParams['voting_scheme'] == 'average':
                predictions = np.mean(predictions_mc,2)
        else:
            for i, (inputs, labels, indices) in enumerate(modelVars['dataloader_'+indices]):
                # Get data
                if mdlParams.get('meta_features',None) is not None: 
                    inputs[0] = inputs[0].cuda()
                    inputs[1] = inputs[1].cuda()
                else:            
                    inputs = inputs.to(modelVars['device'])
                labels = labels.to(modelVars['device'])       
                # Not sure if thats necessary
                modelVars['optimizer'].zero_grad()    
                with torch.set_grad_enabled(False):
                    # Get outputs
                    if mdlParams.get('aux_classifier',False):
                        outputs, outputs_aux = modelVars['model'](inputs)
                        if mdlParams['eval_aux_classifier']:
                            outputs = outputs_aux
                    else:
                        outputs = modelVars['model'](inputs)
                    #print("in",inputs.shape,"out",outputs.shape)
                    preds = modelVars['softmax'](outputs)      
                    # Loss
                    loss = modelVars['criterion'](outputs, labels)     
                # Write into proper arrays                
                if i==0:
                    loss_all = np.array([loss.cpu().numpy()])
                    predictions = preds.cpu().numpy()
                    tar_not_one_hot = labels.data.cpu().numpy()
                    tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                    tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1   
                    targets = tar    
                    #print("Loss",loss_all)         
                else:                 
                    loss_all = np.concatenate((loss_all,np.array([loss.cpu().numpy()])),0)
                    predictions = np.concatenate((predictions,preds.cpu().numpy()),0)
                    tar_not_one_hot = labels.data.cpu().numpy()
                    tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                    tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1                   
                    targets = np.concatenate((targets,tar),0)
                    #allInds[(i*len(mdlParams['numGPUs'])+k)*bSize:(i*len(mdlParams['numGPUs'])+k+1)*bSize] = res_tuple[3][k]
            predictions_mc = predictions
    #print("Check Inds",np.setdiff1d(allInds,mdlParams[indices]))
    # Calculate metrics
    if exclude_class is not None:
        predictions = np.concatenate((predictions[:,:exclude_class],predictions[:,exclude_class+1:]),1)
        targets = np.concatenate((targets[:,:exclude_class],targets[:,exclude_class+1:]),1)    
        num_classes = mdlParams['numClasses']-1
    elif mdlParams['numClasses'] == 9 and mdlParams.get('no_c9_eval',False):
        predictions = predictions[:,:mdlParams['numClasses']-1]
        targets = targets[:,:mdlParams['numClasses']-1]
        num_classes = mdlParams['numClasses']-1
    else:
        num_classes = mdlParams['numClasses']
    # Accuarcy
    acc = np.mean(np.equal(np.argmax(predictions,1),np.argmax(targets,1)))
    # Confusion matrix
    conf = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1))
    if conf.shape[0] < num_classes:
        conf = np.ones([num_classes,num_classes])
    # Class weighted accuracy
    wacc = conf.diagonal()/conf.sum(axis=1)    
    # Sensitivity / Specificity
    sensitivity = np.zeros([num_classes])
    specificity = np.zeros([num_classes])
    if num_classes > 2:
        for k in range(num_classes):
                sensitivity[k] = conf[k,k]/(np.sum(conf[k,:]))
                true_negative = np.delete(conf,[k],0)
                true_negative = np.delete(true_negative,[k],1)
                true_negative = np.sum(true_negative)
                false_positive = np.delete(conf,[k],0)
                false_positive = np.sum(false_positive[:,k])
                specificity[k] = true_negative/(true_negative+false_positive)
                # F1 score
                f1 = f1_score(np.argmax(predictions,1),np.argmax(targets,1),average='weighted')                
    else:
        tn, fp, fn, tp = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1)).ravel()
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        # F1 score
        f1 = f1_score(np.argmax(predictions,1),np.argmax(targets,1))
    # AUC
    fpr = {}
    tpr = {}
    roc_auc = np.zeros([num_classes])
    if num_classes > 9:
        print(predictions)
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(targets[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return np.mean(loss_all), acc, sensitivity, specificity, conf, f1, roc_auc, wacc, predictions, targets, predictions_mc 


def modify_densenet_avg_pool(model):
    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = torch.mean(torch.mean(x,2), 2)
        #x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)

    return model    
    