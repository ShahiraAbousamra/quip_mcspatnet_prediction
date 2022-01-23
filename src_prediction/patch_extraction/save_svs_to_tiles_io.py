import numpy as np
import openslide
import sys
import os
from PIL import Image
import skimage.io as io

slide_name = sys.argv[2] + '/' + sys.argv[1]
output_folder = sys.argv[3] + '/' + sys.argv[1]
#patch_size_20X = 1000
#patch_size_at_target_mag = 1000
#target_mag = 20
#patch_size_at_target_mag = 2000
#target_mag = 12
patch_size_at_target_mag = 1024
target_mag = 20
#target_mag = 40
#mag_40x_magic_number = 40.0 * 0.175  # for til magic number is 10 (0.254 mpp)
mag_40x_magic_number = 10;
fdone = '{}/extraction_done.txt'.format(output_folder)
if os.path.isfile(fdone):
    print('fdone {} exist, skipping'.format(fdone))
    exit(0)

print('extracting {}'.format(output_folder))

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

mag = 20
try:
#    oslide = openslide.OpenSlide(slide_name)
##    mag = mag_40x_magic_number /
##    float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
#    if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
#       mag = mag_40x_magic_number / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
#    elif "XResolution" in oslide.properties:
#       mag = mag_40x_magic_number / float(oslide.properties["XResolution"]);
#    elif 'tiff.XResolution' in oslide.properties:   # for Multiplex IHC WSIs, .tiff images
#       mag = mag_40x_magic_number / float(oslide.properties["tiff.XResolution"]);
#    else:
#       #mag = 10.0 / float(0.254);
#       #mag = mag_40x_magic_number / float(0.175) # 0.175 corresponds to 40x in Dan's data
#       mag = mag_40x_magic_number / float(0.254) # 0.175 corresponds to 40x in Dan's data

    oslide = io.imread(slide_name)
    pw = int(patch_size_at_target_mag * mag / target_mag)
    #pw = int(patch_size_at_target_mag * 0.293 / mpp);
    #pw = 4000
    #patch_size_at_target_mag = pw * mpp / 0.293;
    #width = oslide.dimensions[0]
    #height = oslide.dimensions[1]
    width  = oslide.shape[1]
    height  = oslide.shape[0]
except:
    print('{}: exception caught'.format(slide_name))
    exit(1)


print(slide_name, width, height, pw)

for x in range(1, width, pw):
    for y in range(1, height, pw):
        if x + pw > width:
            pw_x = width - x
        else:
            pw_x = pw
        if y + pw > height:
            pw_y = height - y
        else:
            pw_y = pw

        if (int(patch_size_at_target_mag * pw_x / pw) <= 0) or \
           (int(patch_size_at_target_mag * pw_y / pw) <= 0) or \
           (pw_x <= 0) or (pw_y <= 0):
            continue

        #patch = oslide.read_region((x, y), 0, (pw_x, pw_y))
        patch_arr = oslide[y:y+pw_y, x:x+pw_x]
        if(patch_arr.shape[2] == 4 and patch_arr[:,:,3].max() == 0):
            continue
        #shahira: skip where the alpha channel is zero
        patch = Image.fromarray(patch_arr)
        patch = patch.resize((int(patch_size_at_target_mag * pw_x / pw), int(patch_size_at_target_mag * pw_y / pw)), Image.ANTIALIAS)
        fname = '{}/{}_{}_{}_{}.png'.format(output_folder, x, y, pw, patch_size_at_target_mag)
        patch.save(fname)

open(fdone, 'w').close();

