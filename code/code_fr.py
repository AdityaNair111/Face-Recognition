import numpy as np
import cyvlfeat as vlfeat
from utils import *
import os.path as osp
from glob import glob
from random import shuffle
from IPython.core.debugger import set_trace
from sklearn.svm import LinearSVC
import time
import cv2


def get_positive_features(train_path_pos, feature_params):
    """
    Args:
    -   train_path_pos: (string) This directory contains 36x36 face images
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slower because the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time
            (although you don't have to make the detector step size equal a
            single HoG cell).

    Returns:
    -   feats: N x D matrix where N is the number of faces and D is the template
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    positive_files = glob(osp.join(train_path_pos, '*.jpg'))

    ###########################################################################
    ###########################################################################

    n_cell = np.ceil(win_size/cell_size).astype('int')
    feats=np.zeros((len(positive_files),n_cell*n_cell*31)) # 31 is the default feature dimension
    j=0
    for x in range(0,len(positive_files)):
    	im=load_image_gray(positive_files[x])
    	feats[j,:]=(vlfeat.hog.hog(im, cell_size)).reshape(1,-1)
    	j+=1


    
    #n_cell = np.ceil(win_size/cell_size).astype('int')
    #feats=np.zeros((2*len(positive_files),n_cell*n_cell*31)) # 31 is the default feature dimension
    #j=0
    #for x in range(0,len(positive_files)):
    #	im=load_image_gray(positive_files[x])
    #	feats[j,:]=(vlfeat.hog.hog(im, cell_size)).reshape(1,-1)
    #	j+=1
    #	feats[j,:]=(vlfeat.hog.hog(np.fliplr(im), cell_size)).reshape(1,-1) # positive data augmentation
    #	j+=1 
    	
    	



    ###########################################################################
    ###########################################################################

    return feats

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    """
    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   num_samples: number of negatives to be mined. It is not important for
            the function to find exactly 'num_samples' non-face features. For
            example, you might try to sample some number from each image, but
            some images might be too small to find enough.

    Returns:
    -   N x D matrix where N is the number of non-faces and D is the feature
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    ###########################################################################
    ###########################################################################
    #cell_size=3
    #np.random.seed(seed=5) # for repeating the tests done in parameter tuning
    n_cell = np.ceil(win_size/cell_size).astype('int')
    feats_per_im=(np.ceil(num_samples/len(negative_files))).astype('int')
    feats=np.empty((1,n_cell*n_cell*31))

    counter=0
    for idx, im_filename in enumerate(negative_files):
    	image = load_image_gray(im_filename)
    	temp=vlfeat.hog.hog(image, cell_size)

    	fast_temp=np.zeros((n_cell*(temp.shape[0]-cell_size+1),temp.shape[1],temp.shape[2]))

    	jj=0
    	for ii in range(0,temp.shape[0]-cell_size+1):
    		fast_temp[jj:jj+cell_size,:,:]=temp[ii:ii+cell_size,:,:]
    		jj+=cell_size

    	for jj in range(0,temp.shape[1]-cell_size+1):
    		stack=(fast_temp[:,jj:jj+cell_size,:]).reshape(temp.shape[0]-cell_size+1,-1)
    		r=np.random.randint(0,temp.shape[0]-cell_size,feats_per_im)
    		feats=np.append(feats,stack[r,:],axis=0)
    		counter+=feats_per_im

    		if counter>=num_samples:
    			break

    	if counter>=num_samples:
    			break

    ###########################################################################
    ###########################################################################

    return feats

def train_classifier(features_pos, features_neg, C):
    """

    Args:
    -   features_pos: N X D array. This contains an array of positive features
            extracted from get_positive_feats().
    -   features_neg: M X D array. This contains an array of negative features
            extracted from get_negative_feats().

    Returns:
    -   svm: LinearSVC object. This returns a SVM classifier object trained
            on the positive and negative features.
    """
    ###########################################################################
    ###########################################################################
    svm = LinearSVC(random_state=0,tol=1e-3,loss='hinge',C=C,max_iter=10000)

    N=features_pos.shape[0]
    M=features_neg.shape[0]
    D=features_pos.shape[1]
    train_feats=np.zeros([N+M,D])
    train_feats[:N,:]=features_pos
    train_feats[N:N+M,:]=features_neg
    train_feats.astype('float32')
    y=-np.ones(N+M)
    y[:N]=np.ones(N)

    svm.fit(train_feats,y)

    ###########################################################################
    ###########################################################################

    return svm

def mine_hard_negs(non_face_scn_path, svm, feature_params):
    """
    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features
    -   svm.predict(feat): predict features

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   svm: LinearSVC object

    Returns:
    -   N x D matrix where N is the number of non-faces which are
            false-positive and D is the feature dimensionality.
    """

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    ###########################################################################
    ###########################################################################
    n_cell = np.ceil(win_size/cell_size).astype('int')
    
    counter=0
    scale_f=0
    #scale_f=0.9 # scale factor
    feats=np.empty((1,n_cell*n_cell*31))
    confidence=np.empty(0)

    for idx, im_filename in enumerate(negative_files):
    	image = load_image_gray(im_filename)
    	im_shape=image.shape

    	scales=np.empty((0,1))
    	scaling_percent=1
    	scale=np.array(im_shape)
    	while int(scale[0])>win_size and int(scale[1])>win_size:
    		scales=np.append(scales,scaling_percent)
    		scaling_percent=scaling_percent*scale_f
    		scale=(scaling_percent*np.array(im_shape)).astype("int")


    	for s in scales:
    		temp=(vlfeat.hog.hog(cv2.resize(image,(int(im_shape[1]*s),int(im_shape[0]*s))), cell_size))
    		fast_temp=np.zeros((n_cell*(temp.shape[0]-cell_size+1),temp.shape[1],temp.shape[2]))

    		jj=0
    		for ii in range(temp.shape[0]-cell_size+1):
    			fast_temp[jj:jj+cell_size,:,:]=temp[ii:ii+cell_size,:,:]
    			jj+=cell_size

    		for jj in range(temp.shape[1]-cell_size+1):
    			stack=(fast_temp[:,jj:jj+cell_size,:]).reshape(temp.shape[0]-cell_size+1,-1)
    			v=svm.decision_function(stack)
    			
    			t_test=(v>0) 
    			if np.any(t_test) :
    				y_index=np.nonzero(t_test)[0]
    				passed_stack=stack[y_index,:].reshape(y_index.shape[0],-1)
    				feats=np.append(feats,passed_stack,axis=0)
    				confidence=np.append(confidence,v[y_index])
    				counter+=y_index.shape[0]
    				
    	if counter>=3000:
    		break
    ###########################################################################
    ###########################################################################

    return feats

def run_detector(test_scn_path, svm, feature_params, verbose=False):
    """

    Args:
    -   test_scn_path: (string) This directory contains images which may or
            may not have faces in them. 
    -   svm: A trained sklearn.svm.LinearSVC object
    -   feature_params: dictionary of HoG feature computation parameters.
        Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. 
    -   verbose: prints out debug information if True

    Returns:
    -   bboxes: N x 4 numpy array. N is the number of detections.
            bboxes(i,:) is [x_min, y_min, x_max, y_max] for detection i.
    -   confidences: (N, ) size numpy array. confidences(i) is the real-valued
            confidence of detection i.
    -   image_ids: List with N elements. image_ids[i] is the image file name
            for detection i. (not the full path, just 'albert.jpg')
    """
    
    im_filenames = sorted(glob(osp.join(test_scn_path, '*.jpg')))
    bboxes = np.empty((0, 4))
    confidences = np.empty(0)
    image_ids = []


    # number of top detections to feed to NMS
    topk = 1000
    scale_f=0.9
    tresh=-1.5

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    #scale_factor = feature_params.get('scale_factor', 0.65)
    template_size = int(win_size / cell_size)
    n_cell = np.ceil(win_size/cell_size).astype('int')
    final_val=cell_size - 1
    step_size=1


    for idx, im_filename in enumerate(im_filenames):
        print('Detecting faces in {:s}'.format(im_filename))
        im = load_image_gray(im_filename)
        im_id = osp.split(im_filename)[-1]
        im_shape = im.shape
        # create scale space HOG pyramid and return scores for prediction

        #######################################################################
        #######################################################################

        scales=np.array([1])
        scaling_percent=scale_f
        scale=np.array(im_shape)
        while int(scale[0])>win_size and int(scale[1])>win_size:
            scales=np.append(scales,scaling_percent)
            scaling_percent=scaling_percent*scale_f
            scale=(scaling_percent*np.array(im_shape)).astype("int")
            
        counter=0
        cur_bboxes=np.array([0,0,0,0]).reshape(1,-1) # incase there are zero detection to prevent the code from being stuck
        cur_confidences=np.array([-100]) # very low confidence for this dummy
        for s in scales:
            #print(cell_size/s)
            cur_scale=(vlfeat.hog.hog(cv2.resize(im,(int(im_shape[1]*s),int(im_shape[0]*s))), cell_size))

            fast_scale=np.zeros((int((n_cell*(cur_scale.shape[0]-final_val)/step_size)),cur_scale.shape[1],cur_scale.shape[2]))
            jj=0
            for ii in range(0,cur_scale.shape[0]-final_val,step_size):
            	fast_scale[jj:jj+cell_size,:,:]=cur_scale[ii:ii+cell_size,:,:]
            	jj+=cell_size
            
            for xx in range(0,cur_scale.shape[1]-final_val,step_size):
            	v=svm.decision_function((fast_scale[:,xx:(xx+cell_size),:]).reshape(cur_scale.shape[0]-final_val,-1))
            	if np.any(v>tresh) :
            		y_index=np.nonzero(v>tresh)[0]
            		M=y_index.shape[0]
            		cur_bboxes=np.append(cur_bboxes,((cell_size/s*(np.array([xx*np.ones(M), y_index,(xx+cell_size)*np.ones(M), (y_index+cell_size)]))).astype('int')).T,axis=0)
            		cur_confidences=np.append(cur_confidences,v[np.nonzero(v>tresh)])
            		counter+=M
        #######################################################################
        #######################################################################

        ### non-maximum suppression ###
        # non_max_supr_bbox() can actually get somewhat slow with thousands of
        # initial detections. You could pre-filter the detections by confidence,
        # e.g. a detection with confidence -1.1 will probably never be
        # meaningful. You probably _don't_ want to threshold at 0.0, though. You
        # can get higher recall with a lower threshold. You should not modify
        # anything in non_max_supr_bbox(). If you want to try your own NMS methods,
        # please create another function.

        idsort = np.argsort(-cur_confidences)[:topk]
        cur_bboxes = cur_bboxes[idsort]
        cur_confidences = cur_confidences[idsort]

        is_valid_bbox = non_max_suppression_bbox(cur_bboxes, cur_confidences,
            im_shape, verbose=verbose)

        print('NMS done, {:d} detections passed'.format(sum(is_valid_bbox)))
        cur_bboxes = cur_bboxes[is_valid_bbox]
        cur_confidences = cur_confidences[is_valid_bbox]

        bboxes = np.vstack((bboxes, cur_bboxes))
        confidences = np.hstack((confidences, cur_confidences))
        image_ids.extend([im_id] * len(cur_confidences))

    

    return bboxes, confidences, image_ids
