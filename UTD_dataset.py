#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from sklearn.metrics import confusion_matrix
#import seaborn as sns
#import torch.nn as nn
#import torchvision.models as models
import torchvision
#from tqdm import tqdm
#from sklearn.model_selection import train_test_split
#import matplotlib.patheffects as path_effects
import random
import pickle
#from scipy.stats import wilcoxon
#from scipy.stats import shapiro
#from scipy import stats
#from scipy.stats import ttest_rel
import numpy as np
#from scipy.stats import linregress
#from scipy.interpolate import make_interp_spline, BSpline
#import matplotlib.pyplot as plt
#from sklearn.model_selection import ParameterGrid
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr
#import torch.nn.functional as F
#from torch.autograd import Variable
#from scipy.ndimage.filters import gaussian_filter1d
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from torch.utils.data import DataLoader, random_split
import copy
import math
import shutil


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 0
seed_everything(seed)


# In[2]:


#cl_cr = 0x11 = 3
#cl only = 0x01 = 1
#cr only = 0x10 = 2
#negs = 0
import sys
def load_images_from_dir_image(directory, cl_cr_acces, cl_only_acces, cr_only_acces, cl_cr_neg_acces):
    images_dict = {}
    transform = transforms.Compose([
        transforms.Resize(256),  # first resize the shortest side to 256
        transforms.CenterCrop(224),  # then crop the center 224x224 pixels
        #transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # For pretrain on the Imagenet
    ])
    printed = 0
    for fname in os.listdir(directory):
        if fname.endswith('.jpeg'):
            bag_id = fname.split('_')[0]  # first part before '_' as bag id
            sequence_digits = int(fname.split('_')[2].split(
                '-')[0])  # extract sequence digits

            if bag_id not in images_dict:
                images_dict[bag_id] = {'images': [],
                                       'labels': [], 'sequence_digits': []}

            img_path = os.path.join(directory, fname)
            img = Image.open(img_path).convert('RGB')  # Load and convert image here
            if printed <= 2:
                print("before tx: ", img.size)
                printed = printed+1
            img = transform(img)
            if printed <= 2:
                print("after tx: ", img.size)
                printed = printed+1
            images_dict[bag_id]['images'].append(img)
            if int(bag_id) in cl_cr_acces:
                label = [1,1]
                print("bag_id, label", bag_id, label)
            elif int(bag_id) in cl_only_acces:
                label = [0,1]
                print("bag_id, label", bag_id, label)  
            elif int(bag_id) in cr_only_acces:
                label = [1,0]
                print("bag_id, label", bag_id, label)
            elif int(bag_id) in cl_cr_neg_acces:
                label = [0,0]
                print("bag_id, label", bag_id, label)
            else:
                label = 4
                print("bag_id, label", bag_id, label)
                sys.exit()
            images_dict[bag_id]['labels'].append(label)
            images_dict[bag_id]['sequence_digits'].append(sequence_digits)

    # Sort images by sequence digits within each bag
    for bag_id in images_dict:
        seq_digits, images, labels = zip(*sorted(zip(images_dict[bag_id]['sequence_digits'],
                                                     images_dict[bag_id]['images'],
                                                     images_dict[bag_id]['labels'])))
        images_dict[bag_id]['sequence_digits'] = list(seq_digits)
        images_dict[bag_id]['images'] = list(images)
        images_dict[bag_id]['labels'] = list(labels)

    return images_dict


class MIDataset_image(Dataset):
    """
    A custom dataset class for handling medical images.

    Attributes:
    pos_dict (dict): A dictionary of positive samples.
    neg_dict (dict): A dictionary of negative samples.
    transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
    """
    def __init__(self, pos_neg_dict, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        self.bag_ids = []
        self.sequence_digits = []

        # Combine positive and negative dictionaries
        for bag_id, bag in pos_neg_dict.items():
            self.data.append(bag['images'])  # Store imges directly
            self.labels.append(bag['labels'])
            self.bag_ids.append(bag_id)
            self.sequence_digits.append(bag['sequence_digits'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bag_images = self.data[idx]

        if self.transform:
            bag_images = [self.transform(img) for img in bag_images]

        bag_images = torch.stack(bag_images)
        return bag_images, self.labels[idx], self.bag_ids[idx], self.sequence_digits[idx]


# In[3]:


import os
import pandas as pd
#train_dir = "/home/andrew/aaron/BiSMIL_clr/data_utd/clr_all/train/"
train_dir = "./data_utd/clr_all/train/"
test_dir  = "./data_utd/clr_all/test/"
cl_cr_acces_csv = 'cl_cr_acces.csv'
cl_only_acces_csv = 'cl_only_acces.csv'
cr_only_acces_csv = 'cr_only_acces.csv'
cl_cr_neg_acces_csv = 'cl_cr_neg_acces.csv'

cl_cr_acces = pd.read_csv(cl_cr_acces_csv)
cl_only_acces = pd.read_csv(cl_only_acces_csv)
cr_only_acces = pd.read_csv(cr_only_acces_csv)
cl_cr_neg_acces = pd.read_csv(cl_cr_neg_acces_csv)
cl_cr_acces = cl_cr_acces['accession'].to_list()
cl_only_acces = cl_only_acces['accession'].to_list()
cr_only_acces = cr_only_acces['accession'].to_list()
cl_cr_neg_acces = cl_cr_neg_acces['accession'].to_list()


# In[4]:


# Load images from directories
pos_neg_dict = load_images_from_dir_image(test_dir, cl_cr_acces, cl_only_acces, cr_only_acces, cl_cr_neg_acces)

transform = transforms.Compose([
       #transforms.Resize(256),  # first resize the shortest side to 256
       #transforms.CenterCrop(224),  # then crop the center 224x224 pixels
       transforms.ToTensor(),
       # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # For pretrain on the Imagenet
   ])

test_dataset = MIDataset_image(pos_neg_dict, transform)


# In[5]:


torch.save(test_dataset, './data_utd/UTD_test_dataset.pt')


# In[6]:


transform = transforms.Compose([
        #transforms.Resize(256),  # first resize the shortest side to 256
        #transforms.CenterCrop(244),  # then crop the center 224x224 pixels
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # For pretrain on the Imagenet
    ])
pos_neg_dict = load_images_from_dir_image(train_dir, cl_cr_acces, cl_only_acces, cr_only_acces, cl_cr_neg_acces)


# In[7]:


train_dataset = MIDataset_image(pos_neg_dict, transform)


# In[8]:


torch.save(train_dataset, './data_utd/UTD_train_dataset.pt')


# In[ ]:





# In[ ]:




