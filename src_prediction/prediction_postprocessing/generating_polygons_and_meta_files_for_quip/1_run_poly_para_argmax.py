import os
import time
import glob
import concurrent.futures
import cv2
from get_poly import get_poly
import numpy as np
import sys

#slide_idx = sys.argv[1][0:-len('-multires.tif')]
#pred_suffix = sys.argv[2]

#slide_idx = 'N4277-multires.tif'
#slide_idx = 'O3936-multires.tif'
#slide_idx = 'N9430-multires.tif';
#slide_idx = 'N22034-multires.tif';
#slide_idx = 'O0135-multires.tif';
#slide_idx = 'L6745-multires.tif';
pred_suffix = ''

indir = sys.argv[1];
outdir = sys.argv[2];
slide_idx = sys.argv[3];
slide_idx=slide_idx.rstrip('/')
#slides=os.listdir('/pylon5/ac3uump/shahira/multiplex/u24_lymphocyte/data/patches')
slides=os.listdir(indir)

pred_folder_name = slide_idx+pred_suffix
print('pred_folder_name',pred_folder_name)
print('outdir',outdir)

#infolder='/pylon5/ac3uump/shahira/multiplex/u24_lymphocyte/data/'+pred_folder_name
#save_folder='/pylon5/ac3uump/shahira/multiplex/u24_lymphocyte/data/seg_out/'
#infolder = indir + pred_folder_name
infolder = pred_folder_name
infolder =infolder.rstrip('/')
save_folder = outdir
#argmax_save_folder='../quip4_poly_dots_model_resized/transfered10_300/'
folders=glob.glob(infolder)

cell_type_dict={0:'Lymphocite', 1:'Tumor', 2:'Other'}


make_list = []

folder_i = infolder
folder_i_base = os.path.basename(folder_i)
print('folder_i_base',folder_i_base)
#if not (folder_i_base.startswith('N22034') or folder_i_base.startswith('N9430')):
#    continue

files = glob.glob(os.path.join(folder_i,'*'+'_class_dots.npy'))
#argmax_save=None
#os.path.join(argmax_save_folder[0:-len('/')]+'_argmax_maps',folder_i+'-multires')
#if not os.path.exists(argmax_save):
#    os.makedirs(argmax_save)
for file_i in files:
    make_list.append([file_i,folder_i])
print('make_list',make_list)

'''
for folder_i in folders:
    #debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #selecting slides
    folder_i_base = os.path.basename(folder_i)
    print('folder_i_base',folder_i_base)
    if not (folder_i_base.startswith('N22034') or folder_i_base.startswith('N9430')):
        continue

    files = glob.glob(os.path.join(folder_i,'*'+str(0)+'.png'))
    #argmax_save=None
    #os.path.join(argmax_save_folder[0:-len('/')]+'_argmax_maps',folder_i+'-multires')
    #if not os.path.exists(argmax_save):
    #    os.makedirs(argmax_save)
    for file_i in files:
        make_list.append([file_i,folder_i])
'''
class make_pair_list():
    def __init__(self):
        self.pair_list=[]

    def make_argmax_and_list_for_one(self,pair):
        if True:

            file_i,folder_i = pair
            #argmax_name = os.path.join(folder_i,os.path.basename(file_i)[0:-len('.png')]+'.npy')
            argmax_name = file_i;

            #print('argmax_name',argmax_name)
            #os.path.join(argmax_save,os.path.basename(file_i)[0:-len('_SEG_0.png')]+'.npy')
            #generate argmax map
            '''
            if not os.path.exists(argmax_name):
                print('argmax_name',argmax_name)
                heat_0=cv2.imread(file_i)
                heat_stack=np.zeros((heat_0.shape[0],heat_0.shape[1],8))
                for stain in cell_type_with_BG.keys():
                    heat_i=cv2.imread(file_i[0:-len('0.png')]+str(stain)+'.png',0)
                    heat_stack[:,:,stain] = heat_i
                argmax_map = np.argmax(heat_stack,axis=-1)+1
                np.save(argmax_name,argmax_map)
            '''


            for cell_type_index in cell_type_dict.keys():

                save_path = os.path.join(save_folder,os.path.basename(folder_i),cell_type_dict[cell_type_index])
                print('save_path',save_path)

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                file_prefix=os.path.basename(file_i)[0:-len('_class_dots.npy')]

                if  not os.path.isfile(os.path.join(save_path,file_prefix+'-features.csv')):
                    print(os.path.isfile(os.path.join(save_path,file_prefix+'-features.csv')))
                    print(os.path.join(save_path,file_prefix+'-features.csv'))
                    self.pair_list.append([file_i,save_path,cell_type_index,argmax_name])
                    print('len(self.pair_list)',len(self.pair_list))
                else:
                    print(os.path.join(save_path,file_prefix+'-features.csv'),'exists______________')
                    print(os.path.isfile(os.path.join(save_path,file_prefix+'-features.csv')))

                    #get_poly(file_i,save_path)
def main():
    MAKE_PAIR=make_pair_list()
    #with concurrent.futures.ProcessPoolExecutor( max_workers=10) as executor:
    #    for  prime in  executor.map(MAKE_PAIR.make_argmax_and_list_for_one, make_list, chunksize=10):
    #        print(' is prime: %s' % ( prime))

    print('len(make_list)',len(make_list))
    for pp in make_list:
        MAKE_PAIR.make_argmax_and_list_for_one(pp)
    print('len(pair_list)',len(MAKE_PAIR.pair_list))
    print('len(make_list)',len(make_list))
    with concurrent.futures.ProcessPoolExecutor( max_workers=1) as executor:
        for number, prime in zip(MAKE_PAIR.pair_list, executor.map(get_poly, MAKE_PAIR.pair_list, chunksize=4)):
            print('%s is prime: %s' % (number, prime))


def main_none_para():
    #fid=open(save_folder+'/'+'rm_log.txt','w')
    for pair in pair_list:
        a=get_poly(pair)
if __name__ == '__main__':
    main()
    #main_none_para()
