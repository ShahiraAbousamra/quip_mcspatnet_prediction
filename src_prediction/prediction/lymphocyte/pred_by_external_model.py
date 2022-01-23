import sys
import os
import numpy as np
from PIL import Image

#from external_model_topocount import load_external_model, pred_by_external_model
from external_model_mcspatnet import load_external_model, pred_by_external_model

#APS = 100;
APS = 1024;
#APS = 2000;

TileFolder = sys.argv[1] + '/';
CNNModel = sys.argv[2];
#CNNModel = '/home/shahira/quip_classification/NNFramework_TF/config/test_wsi_ext/config_vgg-mix_test_ext.ini'
#CNNModel = '/home/shahira/quip_classification/NNFramework_TF/config/test_wsi_ext/config_incep-mix_test_ext.ini'
#CNNModel = '/home/shahira/quip_classification/NNFramework_TF/config/test_wsi_ext/config_vgg-mix_test_ext_binary.ini'
#CNNModel = '/home/shahira/quip_classification/NNFramework_TF/config/test_wsi_ext/config_incep-mix_test_ext_binary.ini'

heat_map_out = sys.argv[3];

BatchSize = int(sys.argv[4]); # shahira: Batch size argument
#BatchSize = 96;
#BatchSize = 48;
print('BatchSize = ', BatchSize);
OutputFolder = sys.argv[5];
if(not os.path.exists(OutputFolder)):
    os.mkdir(OutputFolder)

def whiteness(png):
    wh = (np.std(png[:,:,0].flatten()) + np.std(png[:,:,1].flatten()) + np.std(png[:,:,2].flatten())) / 3.0;
    return wh;


def load_data(todo_list, rind):
    X = np.zeros(shape=(BatchSize*40, 3, APS, APS), dtype=np.float32);
    inds = np.zeros(shape=(BatchSize*40,), dtype=np.int32);
    coor = np.zeros(shape=(20000000, 2), dtype=np.int32);

    xind = 0;
    lind = 0;
    cind = 0;
    batch_filenames = [];
    for fn in todo_list:
        lind += 1;
        full_fn = TileFolder + '/' + fn;
        if not os.path.isfile(full_fn):
            continue;
        if len(fn.split('_')) < 4:
            continue;
        if (not fn.endswith('png')):
            continue;

        x_off = float(fn.split('_')[0]);
        y_off = float(fn.split('_')[1]);
        svs_pw = float(fn.split('_')[2]);
        png_pw = float(fn.split('_')[3].split('.png')[0]);

        png = np.array(Image.open(full_fn).convert('RGB'));
        #print('png',png.shape)
        for x in range(0, png.shape[1], APS):
            x_size = APS;
            if x + APS > png.shape[1]:
                #continue;
                x_size = png.shape[1] - x;
            for y in range(0, png.shape[0], APS):
                y_size = APS;
                if y + APS > png.shape[0]:
                    #continue;
                    y_size = png.shape[0] - y;

                ##if (whiteness(png[y:y+APS, x:x+APS, :]) >= 12):
                if (whiteness(png[y:y+y_size, x:x+x_size, :]) >= 12):
                    ##X[xind, :, :, :] = png[y:y+APS, x:x+APS, :].transpose();
                    #X[xind, :, :, :] = png[y:y+APS, x:x+APS, :].transpose((2,0,1));
                    X[xind, :, y:y+y_size, x:x+x_size] = png[y:y+y_size, x:x+x_size, :].transpose((2,0,1));
                    inds[xind] = rind;
                    xind += 1;
                    batch_filenames.append(fn);

                #X[xind, :, y:y+y_size, x:x+x_size] = png[y:y+y_size, x:x+x_size, :].transpose((2,0,1));
                #inds[xind] = rind;
                #xind += 1;
                #batch_filenames.append(fn);

                #coor[cind, 0] = np.int32(x_off + (x + APS/2) * svs_pw / png_pw);
                #coor[cind, 1] = np.int32(y_off + (y + APS/2) * svs_pw / png_pw);
                coor[cind, 0] = np.int32(x_off + (x + x_size/2) * svs_pw / png_pw);
                coor[cind, 1] = np.int32(y_off + (y + y_size/2) * svs_pw / png_pw);
                cind += 1;
                rind += 1;

        if xind >= BatchSize:
            break;

    X = X[0:xind];
    inds = inds[0:xind];
    coor = coor[0:cind];

    return todo_list[lind:], X, inds, coor, rind, batch_filenames;




def val_fn_epoch_on_disk(classn, model):
    all_or = np.zeros(shape=(20000000, classn), dtype=np.float32);
    all_inds = np.zeros(shape=(20000000,), dtype=np.int32);
    all_coor = np.zeros(shape=(20000000, 2), dtype=np.int32);
    rind = 0;
    n1 = 0;
    n2 = 0;
    n3 = 0;
    todo_list = os.listdir(TileFolder);
    # shahira: Handling tensorflow memory exhaust issue on large slides
    reset_limit = 100;
    cur_indx = 0;
    while len(todo_list) > 0:
        todo_list, inputs, inds, coor, rind, batch_filenames = load_data(todo_list, rind);
        if len(inputs) == 0:
            break;
        #print('len(inputs)',len(inputs))
        output = pred_by_external_model(model, inputs);
        #print('len(output)',len(output))

        for patch_indx in range(len(output)):
            #out_filename = TileFolder + '/' + '{}_{}_{}'.format(coor[patch_indx][0], coor[patch_indx][1], APS) + '.npy';
            out_filename = OutputFolder + '/' + os.path.splitext(batch_filenames[patch_indx])[0] + '.npy';            
            output[patch_indx].astype(np.float16).dump(out_filename);

        #all_or[n1:n1+len(output)] = output;
        #all_inds[n2:n2+len(inds)] = inds;
        #all_coor[n3:n3+len(coor)] = coor;
        #n1 += len(output);
        #n2 += len(inds);
        #n3 += len(coor);

        # shahira: Handling tensorflow memory exhaust issue on large slides
        cur_indx += 1;
        if(cur_indx > reset_limit):
            cur_indx = 0;
            #print('Restarting model!');
            #model.restart_model();
            #print('Restarted!');


    #all_or = all_or[:n1];
    #all_inds = all_inds[:n2];
    #all_coor = all_coor[:n3];
    #return all_or, all_inds, all_coor;
    return;


def split_validation(classn):
    model = load_external_model(CNNModel)

    ## Testing
    #Or, inds, coor = val_fn_epoch_on_disk(classn, model);
    #Or_all = np.zeros(shape=(coor.shape[0],), dtype=np.float32);
    #Or_all[inds] = Or[:, 0];

    #fid = open(TileFolder + '/' + heat_map_out, 'w');
    #for idx in range(0, Or_all.shape[0]):
    #    fid.write('{} {} {}\n'.format(coor[idx][0], coor[idx][1], Or_all[idx]));
    #fid.close();
    val_fn_epoch_on_disk(classn, model);
    fid = open(OutputFolder + '/' + heat_map_out, 'w');
    fid.write('\n');
    fid.close();



    return;


def main():
    if not os.path.exists(TileFolder):
        exit(0);

    classes = ['Lymphocytes'];
    classn = len(classes);
    sys.setrecursionlimit(10000);

    split_validation(classn);
    print('DONE!');


if __name__ == "__main__":
    main();
