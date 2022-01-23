import os
import glob
from gen_json import start_json
import sys

#slide_idx = 'N4277-multires.tif';
#slide_idx = 'O3936-multires.tif';
#slide_idx = 'N9430-multires.tif';
#slide_idx = 'N22034-multires.tif';
#slide_idx = 'O0135-multires.tif';
#slide_idx = 'L6745-multires.tif';
#infolder='/gpfs/scratch/sabousamra/multiplex/u24_lymphocyte/data/patches/'
#output_folder = '/gpfs/scratch/sabousamra/multiplex/u24_lymphocyte/data/seg_out/'

infolder = sys.argv[1];
output_folder = sys.argv[2];
print('sys.argv[3]=',sys.argv[3])
slide_idx = os.path.basename(os.path.normpath(sys.argv[3]));
print('slide_idx=',slide_idx)
wsi_folder = sys.argv[4];


#slide_idx = slide_idx[0:-len('')]
slide_folder_suffix = ''
#method_prefix = 'TopoCount-v2-'
#method_prefix = 'TopoCount-v3-'
method_prefix = sys.argv[5];


celltype_num=3
#method_prefix='v7_10X_'
folders=os.listdir(infolder)
#slide_folder_suffix = '_6.6_1.0'

#cell_type={2:'CD3-Double_Negative_T_cell-Yellow',0:'CD16-Myeloid_cell-Black',4:'CD8-Cytotoxic_cell-Purple',1:'CD20-B_cell-Red',3:'CD4-Helper_T_cell-Cyan',5:'K17_Pos',6:'K17_Neg'}
#cell_type={4:'CD3-Yellow', 2:'CD16-Black', 1:'CD8-Purple', 5:'CD20-Red', 3:'CD4-Cyan', 0:'K17_Pos', 7:'K17_Neg', 6:'Hematoxilin'}
cell_type_dict={0:'Lymphocite', 1:'Tumor', 2:'Other'}

print(infolder)
print(output_folder)
print(slide_idx)

for slide_i in folders:
    print('slide_i=',slide_i)
    if not (slide_i.endswith(slide_idx+slide_folder_suffix)):
        print('not (slide_i.endswith(slide_idx+slide_folder_suffix))')
        print('slide_i=',slide_i)
        print('slide_idx=',slide_idx)
        print('slide_folder_suffix=',slide_folder_suffix)
        continue
    print('found')

    #for stain_idx in range(stain_num):
    for cell_type_idx in cell_type_dict.keys():
        png_path= os.path.join(infolder , slide_i,'*'+'_class_dots.npy')
        print('png_path',png_path)
#        print(glob.glob(png_path))
        inpath = os.path.join(infolder,slide_i)
        slide_idx = slide_i.rstrip(slide_folder_suffix)
        save_folder=os.path.join(output_folder,slide_idx ,cell_type_dict[cell_type_idx])
        print('save_folder=',save_folder)
        analysis_id = method_prefix+'-'+cell_type_dict[cell_type_idx]
        print('slide_i,cell_type_idx',slide_i,cell_type_idx)
        print('start_json', slide_i,cell_type_idx,inpath,save_folder,method_prefix,analysis_id,slide_folder_suffix,png_path,wsi_folder)
        start_json(slide_i,cell_type_idx,inpath,save_folder,method_prefix,analysis_id,slide_folder_suffix,png_path,wsi_folder)

        