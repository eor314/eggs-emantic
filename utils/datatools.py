import os
import glob
import numpy as np
from torch.utils.data import Dataset as BaseDataset


class PlanktonDataset(BaseDataset):
    """
    PlanktonDataset. Read images from list in VOC format, apply augmentation and preprocessing transformations.
    :param root: root of VOC-like directory [str]
    :param img_set: path to image list to consider [str] 
    :param augs: transforms to preform for augmentations [func]
    :param preproc: preprocessing steps [resize, normalization] [func]
    :param mapping: dictionary of mask pixel value to class mapping [dict]
    :param dummy_clr: boolean flag to signal stacking gray channel to 3 channel to match network architecture [bool]
    :return: dataset object
    """
    
    CLASSES = ['copepod', 'eggs']
    
    def __init__(self, root=None, img_set=None, classes=None, augs=None, preproc=None, 
                 mapping={50: 1, 100: 2}, dummy_clr=True):

        self.root = root
        img_dir = os.path.join(root, 'JPEGImages')
        seg_dir = os.path.join(root, 'SegmentationMask')
        
        # if the image set given is not an absolute path, assume it lives in VOC structure
        if not os.path.isabs(img_set):
            img_set = os.path.join(root, 'ImageSets', 'Main', img_set)
            
        # get the list of image-ids
        with open(img_set, 'r') as ff:
            tmp = list(ff)
            ff.close
        self.ids = [line.strip() for line in tmp]
        self.images = [os.path.join(img_dir, f'{line}.jpg') for line in self.ids]
        self.masks = [os.path.join(seg_dir, f'{line}.png') for line in self.ids]
        self.class_values = [self.CLASSES.index(ii.lower()) for ii in classes]
        
        self.augs = augs
        self.preproc = preproc
        
        # this is the mapping for the pixel values.
        self.mapping = mapping
    
    def mask_to_class(self, mask):
        for kk in self.mapping:
            mask[mask==kk] = self.mapping[kk]
        return mask
            
    def __getitem__(self, ii):
        
        # read data
        image = cv2.imread(self.images[ii], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks[ii], 0)
        
        # convert mask to numeric labels with the right labels
        mask = self.mask_to_class(mask)
        mks = [(mask==vv+1) for vv in self.class_values]
        mask = np.stack(mks, axis=-1).astype('float')  
        
        # add background if mask not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augs:
            sample = self.augs(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preproc:
            # this sets up dummy color channels in order to do imagenet preprocessing
            if dummy_clf:
                image=np.stack([image, image, image], axis=-1)
                
            sample = self.preproc(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
