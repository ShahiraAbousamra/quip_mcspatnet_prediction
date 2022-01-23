import json
import os
import collections
import sys
import json
import glob
import openslide

def gen_meta_json(in_path, image_id, wsi_width, wsi_height, wsi_mpp, method_description,
        save_folder,Cell_Type):
    file_id = os.path.basename(in_path)[: -len('_class_dots.npy')]
    fields = file_id.split('_')
    x = int(fields[0])
    y = int(fields[1])
    size1 = int(fields[2])
    #size2 = int(fields[3])
    size2 = size1
    #mpp = float(fields[4])
    #mpp = 0.17;
    mpp = wsi_mpp;

    dict_model = collections.OrderedDict()
    dict_model['input_type'] = 'wsi'
    dict_model['otsu_ratio'] = 0.0
    dict_model['curvature_weight'] = 0.0
    dict_model['min_size'] = 1#min_nucleus_size
    dict_model['max_size'] = 5#max_nucleus_size
    dict_model['ms_kernel'] = 0
    dict_model['declump_type'] = 0
    dict_model['levelset_num_iters'] = 0
    dict_model['mpp'] = mpp
    dict_model['image_width'] = wsi_width
    dict_model['image_height'] = wsi_height
    dict_model['tile_minx'] = x
    dict_model['tile_miny'] = y
    dict_model['tile_width'] = size1
    dict_model['tile_height'] = size2
    dict_model['patch_minx'] = x
    dict_model['patch_miny'] = y
    dict_model['patch_width'] = size1
    dict_model['patch_height'] = size2
    dict_model['output_level'] = 'mask'
    dict_model['out_file_prefix'] = file_id
    dict_model['subject_id'] = image_id
    dict_model['case_id'] = image_id
    dict_model['analysis_id'] = Cell_Type
    dict_model['analysis_desc'] = '{}'.format(
            method_description)

    json_str = json.dumps(dict_model)

    fid = open(os.path.join(save_folder, file_id+'-algmeta.json'), 'w')
    print(os.path.join(save_folder, file_id+'-algmeta.json'))
    fid.write(json_str)
    fid.close()

def start_json(image_id,stain_idx,inpath,save_folder,method_prefix,analysis_id,slide_folder_suffix,png_path,wsi_folder):
    if(len(slide_folder_suffix)>0):
        image_id = image_id[0:-len(slide_folder_suffix)]
    print('image_id',image_id)
    method_description = method_prefix
    wsi_path = os.path.join(wsi_folder, image_id+'')
    print('wsi_path',wsi_path)
    oslide = openslide.OpenSlide(wsi_path);
    wsi_width, wsi_height = oslide.dimensions;
    if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
       mpp = float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])
    elif "XResolution" in oslide.properties:
       mpp = float(oslide.properties["XResolution"])
    elif 'tiff.XResolution' in oslide.properties:   # for Multiplex IHC WSIs, .tiff images
       if('tiff.ResolutionUnit' in oslide.properties and oslide.properties["tiff.ResolutionUnit"] == 'centimeter') :
           mpp = 10000/float(oslide.properties["tiff.XResolution"])
       else:
           mpp = float(oslide.properties["tiff.XResolution"])
    else:
       mpp = float(0.254);
       #mpp = float(0.175)
    ##with open('slide_size.json', 'r') as f:
    ##    size_dict = json.load(f)
    ##wsi_width, wsi_height = size_dict[image_id+'-multires.tif']
    files=glob.glob(png_path)
    print('save_folder',save_folder)
    print('png_path',png_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for in_path in files:
        print(in_path)
        gen_meta_json(in_path, image_id+'', wsi_width, wsi_height, mpp, method_description,
                save_folder,analysis_id)
