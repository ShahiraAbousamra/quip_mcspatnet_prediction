import sys;
import os;
import configparser;
import torch;
sys.path.append("..");
sys.path.append(".");
from .mcspatnet_model import MCSpatNetModel

def load_model(model_filepath):

    device_ids_str = None;
    #if(arg_count > min_arg_count):
    #    device_ids_str = sys.argv[2];
    # read the gpu ids to use from the command line parameters if cuda is available
    if(not (device_ids_str is None)):
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str;
        
    device_ids_str = os.environ["CUDA_VISIBLE_DEVICES"];
    print(device_ids_str);

    # read the gpu ids to use from the command line parameters if cuda is available
    device_ids = [];
    device = None;
    if(torch.cuda.is_available()):
        # get number of available cuda devices
        gpu_count = torch.cuda.device_count();
        # create list of gpu ids, excluding invalid gpus
        if(not(device_ids_str is None)):
            device_ids_original = [int(g) for g in device_ids_str.split(",")];            
            device_ids = [j for j in range(len(device_ids_original))];            
            print(device_ids)
            i = 0;
            while(i < len(device_ids)):
                gpu_id = device_ids[i];
                if ((gpu_id >= gpu_count) or (gpu_id < 0)):
                    device_ids.remove(gpu_id);
                else:
                    i = i+1;
            print(device_ids)
            #set the device where data will be placed as the first one in the list
            device = torch.device("cuda:"+str(device_ids[0]) if torch.cuda.is_available() else "cpu");
            print('device_ids[0]' + str(device_ids[0]));
   
    print('device_ids = ' + str(device_ids));
    #if no gpu then cpu
    if(device is None):
        device = torch.device("cpu");
   
    return(MCSpatNetModel(model_filepath, device))

