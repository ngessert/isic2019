import os
import sys
import h5py
import re
import csv
import numpy as np
from glob import glob
import scipy
import pickle
import imagesize

def init(mdlParams_):
    mdlParams = {}
    # Save summaries and model here
    mdlParams['saveDir'] = mdlParams_['pathBase']+'/data/isic/'
    # Data is loaded from here
    mdlParams['dataDir'] = mdlParams_['pathBase']+'/data/isic/2019'

    ### Model Selection ###
    mdlParams['model_type'] = 'efficientnet-b0'
    mdlParams['dataset_names'] = ['official']#,'sevenpoint_rez3_ll']
    mdlParams['file_ending'] = '.png'
    mdlParams['exclude_inds'] = False
    mdlParams['same_sized_crops'] = True
    mdlParams['multiCropEval'] = 9
    mdlParams['var_im_size'] = True
    mdlParams['orderedCrop'] = True
    mdlParams['voting_scheme'] = 'average'    
    mdlParams['classification'] = True
    mdlParams['balance_classes'] = 9
    mdlParams['extra_fac'] = 1.0
    mdlParams['numClasses'] = 9
    mdlParams['no_c9_eval'] = True
    mdlParams['numOut'] = mdlParams['numClasses']
    mdlParams['numCV'] = 5
    mdlParams['trans_norm_first'] = True
    # Scale up for b1-b7
    mdlParams['input_size'] = [224,224,3]    

    ### Training Parameters ###
    # Batch size
    mdlParams['batchSize'] = 20#*len(mdlParams['numGPUs'])
    # Initial learning rate
    mdlParams['learning_rate'] = 0.000015#*len(mdlParams['numGPUs'])
    # Lower learning rate after no improvement over 100 epochs
    mdlParams['lowerLRAfter'] = 25
    # If there is no validation set, start lowering the LR after X steps
    mdlParams['lowerLRat'] = 50
    # Divide learning rate by this value
    mdlParams['LRstep'] = 5
    # Maximum number of training iterations
    mdlParams['training_steps'] = 60 #250
    # Display error every X steps
    mdlParams['display_step'] = 10
    # Scale?
    mdlParams['scale_targets'] = False
    # Peak at test error during training? (generally, dont do this!)
    mdlParams['peak_at_testerr'] = False
    # Print trainerr
    mdlParams['print_trainerr'] = False
    # Subtract trainset mean?
    mdlParams['subtract_set_mean'] = False
    mdlParams['setMean'] = np.array([0.0, 0.0, 0.0])   
    mdlParams['setStd'] = np.array([1.0, 1.0, 1.0])   

    # Data AUG
    #mdlParams['full_color_distort'] = True
    mdlParams['autoaugment'] = False
    mdlParams['flip_lr_ud'] = True
    mdlParams['full_rot'] = 180
    mdlParams['scale'] = (0.8,1.2)
    mdlParams['shear'] = 10
    mdlParams['cutout'] = 16

    # Meta settings
    mdlParams['meta_features'] = ['age_num','sex_oh','loc_oh']
    mdlParams['meta_feature_sizes'] = [1,8,2]
    mdlParams['encode_nan'] = False
    # Pretrained model from task 1
    mdlParams['model_load_path'] = mdlParams_['pathBase']+'/data/isic/2019.test_effb0_ss'
    mdlParams['fc_layers_before'] = [256,256]
    # Factor for scaling up the FC layer
    scale_up_with_larger_b = 1.0
    mdlParams['fc_layers_after'] = [int(1024*scale_up_with_larger_b)]
    mdlParams['freeze_cnn'] = True
    mdlParams['learning_rate_meta'] = 0.00001
    # each feature is set to missing with this prob
    mdlParams['drop_augment'] = 0.1
    # Normal dropout in fc layers
    mdlParams['dropout_meta'] = 0.4
    mdlParams['scale_features'] = True      

    ### Data ###
    mdlParams['preload'] = False
    # Labels first
    # Targets, as dictionary, indexed by im file name
    mdlParams['labels_dict'] = {}
    path1 = mdlParams['dataDir'] + '/labels/'
     # All sets
    allSets = glob(path1 + '*/')   
    # Go through all sets
    for i in range(len(allSets)):
        # Check if want to include this dataset
        foundSet = False
        for j in range(len(mdlParams['dataset_names'])):
            if mdlParams['dataset_names'][j] in allSets[i]:
                foundSet = True
        if not foundSet:
            continue                
        # Find csv file
        files = sorted(glob(allSets[i]+'*'))
        for j in range(len(files)):
            if 'csv' in files[j]:
                break
        # Load csv file
        with open(files[j], newline='') as csvfile:
            labels_str = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in labels_str:
                if 'image' == row[0]:
                    continue
                #if 'ISIC' in row[0] and '_downsampled' in row[0]:
                #    print(row[0])
                if row[0] + '_downsampled' in mdlParams['labels_dict']:
                    print("removed",row[0] + '_downsampled')
                    continue
                if mdlParams['numClasses'] == 7:
                    mdlParams['labels_dict'][row[0]] = np.array([int(float(row[1])),int(float(row[2])),int(float(row[3])),int(float(row[4])),int(float(row[5])),int(float(row[6])),int(float(row[7]))])
                elif mdlParams['numClasses'] == 8:
                    if len(row) < 9 or row[8] == '':
                        class_8 = 0
                    else:
                        class_8 = int(float(row[8]))
                    mdlParams['labels_dict'][row[0]] = np.array([int(float(row[1])),int(float(row[2])),int(float(row[3])),int(float(row[4])),int(float(row[5])),int(float(row[6])),int(float(row[7])),class_8])
                elif mdlParams['numClasses'] == 9:
                    if len(row) < 9 or row[8] == '':
                        class_8 = 0
                    else:
                        class_8 = int(float(row[8]))  
                    if len(row) < 10 or row[9] == '':
                        class_9 = 0
                    else:
                        class_9 = int(float(row[9]))                                           
                    mdlParams['labels_dict'][row[0]] = np.array([int(float(row[1])),int(float(row[2])),int(float(row[3])),int(float(row[4])),int(float(row[5])),int(float(row[6])),int(float(row[7])),class_8,class_9])
    # Load meta data
    mdlParams['meta_dict'] = {}
    path1 = mdlParams['dataDir'] + '/meta_data/'
     # All sets
    allSets = glob(path1 + '*/')   
    # Go through all sets
    for i in range(len(allSets)):
        # Check if want to include this dataset
        foundSet = False
        for j in range(len(mdlParams['dataset_names'])):
            if mdlParams['dataset_names'][j] in allSets[i]:
                foundSet = True
        if not foundSet:
            continue                
        # Find csv file
        files = sorted(glob(allSets[i]+'*'))
        for j in range(len(files)):
            if '.pkl' in files[j]:
                break    
        # Open and load
        with open(files[j],'rb') as f:
            meta_data = pickle.load(f)
        # Write into dict
        for k in range(len(meta_data['im_name'])):
            feature_vector = []
            if 'age_oh' in mdlParams['meta_features']:
                if mdlParams['encode_nan']:
                    feature_vector.append(meta_data['age_oh'][k,:])
                else:
                    feature_vector.append(meta_data['age_oh'][k,1:])
            if 'age_num' in mdlParams['meta_features']:
                feature_vector.append(np.array([meta_data['age_num'][k]]))                      
            if 'loc_oh' in mdlParams['meta_features']:
                if mdlParams['encode_nan']:
                    feature_vector.append(meta_data['loc_oh'][k,:])
                else:
                    feature_vector.append(meta_data['loc_oh'][k,1:])
            if 'sex_oh' in mdlParams['meta_features']:
                if mdlParams['encode_nan']:
                    feature_vector.append(meta_data['sex_oh'][k,:])
                else:
                    feature_vector.append(meta_data['sex_oh'][k,1:]) 

            #print(feature_vector) 
            feature_vector = np.concatenate(feature_vector,axis=0)
            #print("feature vector shape",feature_vector.shape)                                                
            mdlParams['meta_dict'][meta_data['im_name'][k]] = feature_vector    


    # Save all im paths here
    mdlParams['im_paths'] = []
    mdlParams['labels_list'] = []
    mdlParams['meta_list'] = [] 
    # Define the sets
    path1 = mdlParams['dataDir'] + '/images/'
    # All sets
    allSets = sorted(glob(path1 + '*/'))
    # Ids which name the folders
    # Make official first dataset
    for i in range(len(allSets)):
        if mdlParams['dataset_names'][0] in allSets[i]:
            temp = allSets[i]
            allSets.remove(allSets[i])
            allSets.insert(0, temp)
    print(allSets)        
    # Set of keys, for marking old HAM10000
    mdlParams['key_list'] = []
    if mdlParams['exclude_inds']:
        with open(mdlParams['saveDir'] + 'indices_exclude.pkl','rb') as f:
            indices_exclude = pickle.load(f)          
        exclude_list = []    
    for i in range(len(allSets)):
        # All files in that set
        files = sorted(glob(allSets[i]+'*'))
        # Check if there is something in there, if not, discard
        if len(files) == 0:
            continue
        # Check if want to include this dataset
        foundSet = False
        for j in range(len(mdlParams['dataset_names'])):
            if mdlParams['dataset_names'][j] in allSets[i]:
                foundSet = True
        if not foundSet:
            continue                    
        for j in range(len(files)):
            if '.jpg' in files[j] or '.jpeg' in files[j] or '.JPG' in files[j] or '.JPEG' in files[j] or '.png' in files[j] or '.PNG' in files[j]:                
                # Add according label, find it first
                found_already = False
                for key in mdlParams['labels_dict']:
                    if key + mdlParams['file_ending'] in files[j]:
                        if found_already:
                            print("Found already:",key,files[j])                     
                        mdlParams['key_list'].append(key)
                        mdlParams['labels_list'].append(mdlParams['labels_dict'][key])
                        mdlParams['meta_list'].append(mdlParams['meta_dict'][key])
                        found_already = True
                if found_already:
                    mdlParams['im_paths'].append(files[j])     
                    if mdlParams['exclude_inds']:
                        for key in indices_exclude:
                            if key in files[j]:
                                exclude_list.append(indices_exclude[key])                                       
    # Convert label list to array
    mdlParams['labels_array'] = np.array(mdlParams['labels_list'])
    print(np.mean(mdlParams['labels_array'],axis=0))      
    # Meta data
    mdlParams['meta_array'] = np.array(mdlParams['meta_list'])
    print("final meta shape",mdlParams['meta_array'].shape)        
    # Create indices list with HAM10000 only
    mdlParams['HAM10000_inds'] = []
    HAM_START = 24306
    HAM_END = 34320
    for j in range(len(mdlParams['key_list'])):
        try:
            curr_id = [int(s) for s in re.findall(r'\d+',mdlParams['key_list'][j])][-1]
        except:
            continue
        if curr_id >= HAM_START and curr_id <= HAM_END:
            mdlParams['HAM10000_inds'].append(j)
    mdlParams['HAM10000_inds'] = np.array(mdlParams['HAM10000_inds'])    
    print("Len ham",len(mdlParams['HAM10000_inds']))   
    # Perhaps preload images
    if mdlParams['preload']:
        mdlParams['images_array'] = np.zeros([len(mdlParams['im_paths']),mdlParams['input_size_load'][0],mdlParams['input_size_load'][1],mdlParams['input_size_load'][2]],dtype=np.uint8)
        for i in range(len(mdlParams['im_paths'])):
            x = scipy.ndimage.imread(mdlParams['im_paths'][i])
            #x = x.astype(np.float32)   
            # Scale to 0-1 
            #min_x = np.min(x)
            #max_x = np.max(x)
            #x = (x-min_x)/(max_x-min_x)
            mdlParams['images_array'][i,:,:,:] = x
            if i%1000 == 0:
                print(i+1,"images loaded...")     
    if mdlParams['subtract_set_mean']:
        mdlParams['images_means'] = np.zeros([len(mdlParams['im_paths']),3])
        for i in range(len(mdlParams['im_paths'])):
            x = scipy.ndimage.imread(mdlParams['im_paths'][i])
            x = x.astype(np.float32)   
            # Scale to 0-1 
            min_x = np.min(x)
            max_x = np.max(x)
            x = (x-min_x)/(max_x-min_x)
            mdlParams['images_means'][i,:] = np.mean(x,(0,1))
            if i%1000 == 0:
                print(i+1,"images processed for mean...")         

    ### Define Indices ###
    with open(mdlParams['saveDir'] + 'indices_isic2019.pkl','rb') as f:
        indices = pickle.load(f)            
    mdlParams['trainIndCV'] = indices['trainIndCV']
    mdlParams['valIndCV'] = indices['valIndCV']
    if mdlParams['exclude_inds']:
        exclude_list = np.array(exclude_list)
        all_inds = np.arange(len(mdlParams['im_paths']))
        exclude_inds = all_inds[exclude_list.astype(bool)]
        for i in range(len(mdlParams['trainIndCV'])):
            mdlParams['trainIndCV'][i] = np.setdiff1d(mdlParams['trainIndCV'][i],exclude_inds)
        for i in range(len(mdlParams['valIndCV'])):
            mdlParams['valIndCV'][i] = np.setdiff1d(mdlParams['valIndCV'][i],exclude_inds)     
    # Consider case with more than one set
    if len(mdlParams['dataset_names']) > 1:
        restInds = np.array(np.arange(25331,mdlParams['labels_array'].shape[0]))
        for i in range(mdlParams['numCV']):
            mdlParams['trainIndCV'][i] = np.concatenate((mdlParams['trainIndCV'][i],restInds))        
    print("Train")
    for i in range(len(mdlParams['trainIndCV'])):
        print(mdlParams['trainIndCV'][i].shape)
    print("Val")
    for i in range(len(mdlParams['valIndCV'])):
        print(mdlParams['valIndCV'][i].shape)    

    # Use this for ordered multi crops
    if mdlParams['orderedCrop']:
        # Crop positions, always choose multiCropEval to be 4, 9, 16, 25, etc.
        mdlParams['cropPositions'] = np.zeros([len(mdlParams['im_paths']),mdlParams['multiCropEval'],2],dtype=np.int64)
        #mdlParams['imSizes'] = np.zeros([len(mdlParams['im_paths']),mdlParams['multiCropEval'],2],dtype=np.int64)
        for u in range(len(mdlParams['im_paths'])):
            height, width = imagesize.get(mdlParams['im_paths'][u])
            if width < mdlParams['input_size'][0]:
                height = int(mdlParams['input_size'][0]/float(width))*height
                width = mdlParams['input_size'][0]
            if height < mdlParams['input_size'][0]:
                width = int(mdlParams['input_size'][0]/float(height))*width
                height = mdlParams['input_size'][0]            
            ind = 0
            for i in range(np.int32(np.sqrt(mdlParams['multiCropEval']))):
                for j in range(np.int32(np.sqrt(mdlParams['multiCropEval']))):
                    mdlParams['cropPositions'][u,ind,0] = mdlParams['input_size'][0]/2+i*((width-mdlParams['input_size'][1])/(np.sqrt(mdlParams['multiCropEval'])-1))
                    mdlParams['cropPositions'][u,ind,1] = mdlParams['input_size'][1]/2+j*((height-mdlParams['input_size'][0])/(np.sqrt(mdlParams['multiCropEval'])-1))
                    #mdlParams['imSizes'][u,ind,0] = curr_im_size[0]

                    ind += 1
        # Sanity checks
        #print("Positions",mdlParams['cropPositions'])
        # Test image sizes
        height = mdlParams['input_size'][0]
        width = mdlParams['input_size'][1]
        for u in range(len(mdlParams['im_paths'])):
            height_test, width_test = imagesize.get(mdlParams['im_paths'][u])
            if width_test < mdlParams['input_size'][0]:
                height_test = int(mdlParams['input_size'][0]/float(width_test))*height_test
                width_test = mdlParams['input_size'][0]
            if height_test < mdlParams['input_size'][0]:
                width_test = int(mdlParams['input_size'][0]/float(height_test))*width_test
                height_test = mdlParams['input_size'][0]                
            test_im = np.zeros([width_test,height_test]) 
            for i in range(mdlParams['multiCropEval']):
                im_crop = test_im[np.int32(mdlParams['cropPositions'][u,i,0]-height/2):np.int32(mdlParams['cropPositions'][u,i,0]-height/2)+height,np.int32(mdlParams['cropPositions'][u,i,1]-width/2):np.int32(mdlParams['cropPositions'][u,i,1]-width/2)+width]
                if im_crop.shape[0] != mdlParams['input_size'][0]:
                    print("Wrong shape",im_crop.shape[0],mdlParams['im_paths'][u])    
                if im_crop.shape[1] != mdlParams['input_size'][1]:
                    print("Wrong shape",im_crop.shape[1],mdlParams['im_paths'][u])        
    return mdlParams