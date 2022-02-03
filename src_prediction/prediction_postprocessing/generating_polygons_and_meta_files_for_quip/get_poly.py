import cv2
import subprocess
import os
import numpy as np
#file_name='/scratch/KurcGroup/mazhao/wsi_tiles_prediction/O3936-multires/96001_72001_4000_4000_0.25_1_SEG_0_pred.png'
#save_path='/scratch/KurcGroup/mazhao/quip4_files/'+os.path.basename(os.path.dirname(file_name)+'/'+cell_type[stain_num])
#save_path='.'
cell_indx_to_id={0:1  , 1:2, 2:3}
def get_poly(pair):
        thre_mode = 0
        print('len(pair)',len(pair))
        file_name,save_path,class_index,argmax_name = pair
        print(pair)
        if argmax_name==None:
            thre_mode = 1
        else:
            print('argmax mode!')
        #file_name is the heatmap absolute path,save_path is the folder to save the result
        #if not os.path.isfile(os.path.join(save_path,file_id+'-features.csv')):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        global_xy_offset= [int(x) for x in os.path.basename(file_name).split('_')[0:2]]
        original_patch_size = int(os.path.basename(file_name).split('_')[2])
        if thre_mode==1:
            img=cv2.imread(file_name,0)
            print('file_name',file_name)
            thre,img=cv2.threshold(img,210,255,cv2.THRESH_BINARY)
        else:
            print(argmax_name)
            argmax_map = np.load(argmax_name, allow_pickle=True)
            binary_mask = np.zeros((argmax_map.shape[-1],argmax_map.shape[-2])).astype('uint8')
            binary_mask[argmax_map[class_index] == 1]=255
            #cv2.imwrite('/gpfs/scratch/sabousamra/multiplex/u24_lymphocyte/data/seg_out/tmp/binary_'+os.path.basename(file_name)+str(stain_index)+'.png',binary_mask)
            img = binary_mask
            scale = original_patch_size/max(img.shape[0], img.shape[1])

            #resizing to 2 times!!!!!!!!!!!!!!!!!
            #heat_map = cv2.imread(file_name)
            #img = cv2.resize(img,(heat_map.shape[1],heat_map.shape[0]),cv2.INTER_NEAREST)
            img = cv2.resize(img,(int(scale*img.shape[1]), int(scale*img.shape[0])),cv2.INTER_NEAREST)
            #cv2.imwrite(os.path.join(save_path,os.path.basename(file_name)[0:-10]+'-binary.png'),img)

        #print(np.max(img),img.shape)
        #cv2.imwrite('/gpfs/scratch/sabousamra/multiplex/u24_lymphocyte/data/seg_out/tmp/binary_'+os.path.basename(file_name)+str(stain_index)+'xx.png',binary_mask)
        file_id=os.path.basename(file_name)[0:-len('_class_dots.npy')]
        fid = open(os.path.join(save_path,file_id+'-features.csv'), 'w')
        fid.write('AreaInPixels,PhysicalSize,ClassId,Polygon\n')
        #binary_mask = (binary_mask>0).astype('uint8');
        poly = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #poly = cv2.findContours(binary_mask.astype('uint8'), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        print('len(poly)=',len(poly))
        if(len(poly) == 2):
            contours,hia = poly
        else:
            im, contours,hia = poly
        #img_out = cv2.imread(os.path.join(os.path.split(file_name)[0],file_id+'.png'))
        #img_out = cv2.resize(img_out ,(int(img_out.shape[1]*2883/1000),int(img_out.shape[0]*2883/1000)),cv2.INTER_NEAREST)
        contour = contours
        num_contour=len(contour)
        print('num_contour=',num_contour);
        for idx in range(num_contour):
            contour_i = contour[idx]
            physical_size = cv2.contourArea(contour_i)
            #print(physical_size)
            #if physical_size>4000 or physical_size<85:
            #if physical_size<85:
            #    continue
            #cv2.drawContours(img_out, contours, idx, (0,255,0), 2)
            contour_i = contour_i[:,0,:].astype(np.float32)

            contour_i[:, 0] = contour_i[:, 0] + global_xy_offset[0]

            contour_i[:, 1] = contour_i[:, 1]  + global_xy_offset[1]
            poly_str = ':'.join(['{:.1f}'.format(x) for x in contour_i.flatten().tolist()])
            #print(poly_str)
            #fid.write('{},{},[{}]\n'.format(int(physical_size), int(physical_size), poly_str))
            fid.write('{},{},{},[{}]\n'.format(int(physical_size), int(physical_size), cell_indx_to_id[class_index], poly_str))
        fid.close()
        #cv2.imwrite(os.path.join('/gpfs/scratch/sabousamra/multiplex/u24_lymphocyte/data/seg_out/tmp',file_id+str(stain_index)+'.png'),img_out);
        return 1
