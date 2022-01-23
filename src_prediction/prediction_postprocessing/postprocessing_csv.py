import numpy as np
import os
import glob
import skimage.io as io
from skimage import measure
import cv2
import sys
from skimage import filters
from scipy.ndimage.filters import convolve    
from skimage.measure import label, moments

in_dir = sys.argv[2] + '/' + sys.argv[1]
out_dir = sys.argv[3] + '/' + sys.argv[1]
out_dir_csv = sys.argv[4] + '/' + sys.argv[1]
fdone = '{}/postprocessing_done.txt'.format(out_dir)
if os.path.isfile(fdone):
    print('fdone {} exist, skipping'.format(fdone))
    exit(0)

print('postprocessing {}'.format(out_dir))

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if not os.path.exists(out_dir_csv):
    os.mkdir(out_dir_csv)

 
#thresh_low = 0.8
#thresh_high = 0.9
thresh_low = 0.5
thresh_high = 0.5
#size_thresh = -1
n_classes = 4
vis_kernel = np.ones((7,7))

et_dmap_files = glob.glob(os.path.join(in_dir, '*.npy'))

#print('predictionfiles', len(et_dmap_files))
for file in et_dmap_files:
    #print(file)
    et_dmap = np.load(file, allow_pickle=True)
    #img_path = file[:-len('.npy')] + '.png'
    #out_class_softmax_filepath = os.path.join(out_dir, os.path.splitext(os.path.basename(file))[0]) + '_class_softmax.npy'
    out_class_centers_filepath = os.path.join(out_dir, os.path.splitext(os.path.basename(file))[0]) + '_class_centers.npy'
    out_class_dots_filepath = os.path.join(out_dir, os.path.splitext(os.path.basename(file))[0]) + '_class_dots.npy'
    out_all_centers_filepath = os.path.join(out_dir, os.path.splitext(os.path.basename(file))[0]) + '_all_centers.npy'
    out_all_dots_filepath = os.path.join(out_dir, os.path.splitext(os.path.basename(file))[0]) + '_all_dots.npy'
    
    if(os.path.isfile(out_class_centers_filepath) and os.path.isfile(out_class_dots_filepath) and os.path.isfile(out_all_centers_filepath) and os.path.isfile(out_all_dots_filepath )):
        continue
    et_class_sig = et_dmap[1:,:,:]
    et_all_sig = et_dmap[0,:,:]
    #et_dmap_class = et_dmap.detach().cpu().numpy()
    #et_dmap_all = et_dmap.detach().cpu().numpy()
    
    e_hard = filters.apply_hysteresis_threshold(et_all_sig.squeeze(), thresh_low, thresh_high)            
    e_hard2 = (e_hard > 0).astype(np.uint8)
    e_hard2_all = e_hard2.copy() 
    comp_mask = label(e_hard2)
    #e_count = comp_mask.max()
    #s_count=0
    #if(size_thresh > 0):
    #    for c in range(1,comp_mask.max()+1):
    #        s = (comp_mask == c).sum()
    #        if(s < size_thresh):
    #            e_count -=1
    #            s_count +=1

    e_dot_all = np.zeros((e_hard.shape[0], e_hard.shape[1]))
    e_dot_all_vis = np.zeros((e_hard.shape[0], e_hard.shape[1]))
    contours, hierarchy = cv2.findContours(e_hard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for idx in range(len(contours)):
        #print('idx=',idx)
        contour_i = contours[idx]
        M = cv2.moments(contour_i)
        #print(M)
        if(M['m00'] == 0):
            continue;
        cx = round(M['m10'] / M['m00'])
        cy = round(M['m01'] / M['m00'])
        e_dot_all_vis[max(0,cy-vis_kernel.shape[0]//2):min(cy+vis_kernel.shape[0]//2+1,e_dot_all_vis.shape[0]-1), max(0,cx-vis_kernel.shape[1]//2):min(cx+vis_kernel.shape[1]//2+1,e_dot_all_vis.shape[1]-1)] = 1
        e_dot_all[cy, cx] = 1

                
    e_dot_all.astype(np.uint8).dump(out_all_centers_filepath)
    #e_dot_all_vis.astype(np.uint8).dump(out_all_dots_filepath)
    #points = np.where(e_dot_all)
    #if(len(points[0]) > 0):
    #    points_arr=np.zeros((len(points[0]),2), dtype=int)
    #    points_arr[:,0] = points[1]
    #    points_arr[:,1] = points[0]
    #    np.savetxt(os.path.join(out_dir_csv, os.path.splitext(os.path.basename(file))[0]) + '_class_centers.npy', points_arr,newline='\n', delimiter=' ', fmt='%d')

    e_dot_class = np.zeros((n_classes-1, e_dot_all.shape[0], e_dot_all.shape[1]))
    e_dot_class_vis = np.zeros((n_classes-1, e_dot_all.shape[0], e_dot_all.shape[1]))
    et_class_argmax = et_class_sig.squeeze().argmax(axis=0)
    for s in range(n_classes-1):
        #e_hard = filters.apply_hysteresis_threshold(et_class_sig[s].squeeze(), thresh_low, thresh_high)
        #e_hard2 = (e_hard > 0).astype(np.uint8)               
        e_hard2 = (et_class_argmax == s)  

        e_dot_class[s] = e_hard2 * e_dot_all  
        #e_count = e_dot.sum()  
        e_dot_class_vis[s] = (convolve(e_dot_class[s], vis_kernel) > 0).astype(np.uint8)               
    e_dot_class.astype(np.uint8).dump(out_class_centers_filepath)
    e_dot_class_vis.astype(np.uint8).dump(out_class_dots_filepath)
    for s in range(n_classes-1):
        io.imsave(out_class_dots_filepath.replace('.npy', '_s'+str(s)+'.png'), (e_dot_class_vis[s]*255).astype(np.uint8))
        points = np.where(e_dot_class[s])
        if(len(points[0]) > 0):
            points_arr=np.zeros((len(points[0]),2), dtype=int)
            points_arr[:,0] = points[1]
            points_arr[:,1] = points[0]
            np.savetxt(os.path.join(out_dir_csv, os.path.splitext(os.path.basename(file))[0]) + '_class_centers_s'+str(s)+'.csv', points_arr,newline='\n', delimiter=' ', fmt='%d')
    


open(fdone, 'w').close();

