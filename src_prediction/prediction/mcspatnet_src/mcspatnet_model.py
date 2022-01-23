#import tensorflow as tf;
#import torch.optim as optim
import os;
from distutils.util import strtobool;
import numpy as np;
import torch
import glob;
import skimage.io as io;
import torch
import torch.nn as nn

from .model_arch import UnetVggMultihead


class MCSpatNetModel:
    def __init__(self, model_filepath, device, kwargs={}):
        # predefined list of arguments
        args = {};
        args.update(kwargs);

        self.model_filepath = model_filepath
        self.device = device

        self.dropout_keep_prob = 1.0
        self.initial_pad = 126
        self.interpolate = 'False'
        self.conv_init = 'he'
        self.n_classes = 4
        self.class_indx = '1,2,3'  
        self.class_weights = np.array([1,1,1]) 
        self.thresh_low = 0.8
        self.thresh_high = 0.9
        self.size_thresh = -1
        self.max_scale = 16

        gpu_or_cpu='cuda' # use cuda or cpu
        dropout_prob = 0
        initial_pad = 126
        interpolate = 'False'
        conv_init = 'he'
        n_classes = 3
        n_classes_out = n_classes + 1
        class_indx = '1,2,3'
        class_weights = np.array([1,1,1]) 
        n_clusters = 5
        n_classes2 = n_clusters * (n_classes)

        r_step = 15
        r_range = range(0, 100, r_step)
        r_arr = np.array([*r_range])
        r_classes = len(r_range)
        r_classes_all = r_classes * (n_classes )

        self.model=UnetVggMultihead(kwargs={'dropout_prob':dropout_prob, 'initial_pad':initial_pad, 'interpolate':interpolate, 'conv_init':conv_init, 'n_classes':n_classes, 'n_channels':3, 'n_heads':4, 'head_classes':[1,n_classes,n_classes2, r_classes_all]})
        if(not (model_filepath is None)):
            self.model.load_state_dict(torch.load(model_filepath), strict=True);
        self.model.to(device)
        self.criterion_sig = nn.Sigmoid() # initialize sigmoid layer
        self.criterion_softmax = nn.Softmax(dim=1) # initialize sigmoid layer




    def predict(self, inputs):
        with torch.no_grad():
            batch_x = inputs;
            if (batch_x is None):
                return None;
            
            batch_x = self.preprocess_input(inputs);

            if(self.device is not None):
                batch_x = batch_x.to(self.device);

            et_dmap_lst=self.model(batch_x)
            et_dmap_all=et_dmap_lst[0][:,:,2:-2,2:-2]
            et_dmap_class=et_dmap_lst[1][:,:,2:-2,2:-2]

            et_all_sig = self.criterion_sig(et_dmap_all).detach().cpu().numpy().astype(np.float16);
            et_class_sig = self.criterion_softmax(et_dmap_class).detach().cpu().numpy().astype(np.float16);
            et_dmap = np.concatenate((et_all_sig, et_class_sig), axis=1)



            return et_dmap;


  
    def preprocess_input(self, inputs):        
        # inputs format is (batch, channel, y, x) and range (0,255)
        inputs /= 255            
        if self.max_scale>1: # to downsample image and density-map to match deep-model.
            ds_rows=int(inputs.shape[-2]//self.max_scale)*self.max_scale
            ds_cols=int(inputs.shape[-1]//self.max_scale)*self.max_scale
            pad_y1 = 0
            pad_y2 = 0
            pad_x1 = 0
            pad_x2 = 0
            if(ds_rows < inputs.shape[-2]):
                pad_y1 = (self.max_scale - (inputs.shape[-2] - ds_rows))//2
                pad_y2 = (self.max_scale - (inputs.shape[-2] - ds_rows)) - pad_y1
            if(ds_cols < inputs.shape[-1]):
                pad_x1 = (self.max_scale - (inputs.shape[-1] - ds_cols))//2
                pad_x2 = (self.max_scale - (inputs.shape[-1] - ds_cols)) - pad_x1
            inputs = np.pad(inputs,((0,0),(0,0), (pad_y1,pad_y2),(pad_x1,pad_x2)), 'constant', constant_values=(1,) )# padding constant differs by dataset based on bg color


        inputs = torch.tensor(inputs, dtype = torch.float); # to avoid the error:  Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same

        return inputs;