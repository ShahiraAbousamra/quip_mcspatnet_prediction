# Quip MCSpatNet Cell Prediction

###Code for whole slide image prediction using our model MCSpatNet ([Multi-Class Cell Detection Using Spatial Context Representation, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Abousamra_Multi-Class_Cell_Detection_Using_Spatial_Context_Representation_ICCV_2021_paper.pdf)) 


#### Required settings in `src_prediction/conf/variables.sh`

`LYM_PREDICTION_BATCH_SIZE`: The batch size to use.  <br/>
`BASE_DIR`=< The full path to the `u24_lymphocyte` directory > <br/>
`OUT_DIR`=< The full path to the output directory > <br/>
`PREDICTION_INTERMEDIATE_FOLDER`=< The full path for intermediate patch results > <br/>
`POSTPROCESS_DIR`=< The full path for processed patch results > <br/>
`JSON_OUTPUT_FOLDER`=< The full path for final quip format patch results > <br/>
`SVS_INPUT_PATH`=< The full path to the WSI files > <br/>
`PATCH_PATH`=< The full path to the output from WSI patch extraction > <br/>
`LYM_NECRO_CNN_MODEL_PATH`=< The full path to trained MCSpatNet model >  <br/>
`CSV_OUTPUT_FOLDER`=< The full path for results in csv format > <br/>

#### To run:<br/>
`cd src_prediction/scripts` <br/> 
`CUDA_VISIBLE_DEVICES='0' nohup bash svs_2_heatmap.sh &` <br/>

#### Environment <br/>
Python >= 3.6 <br/>
Pytorch >= 1.0 <br/>
Openslide <br/>
numpy <br/>
OpenCV <br/>

#### Conda Environment <br/>
conda create --name wsi-pytorch -c pytorch -c conda-forge python=3.7 pytorch torchvision torchaudio cudatoolkit=11.3 openslide openslide-python pandas scikit-learn scikit-image opencv

### Citation ###
If you find this code helpful, please cite our paper:

	@InProceedings{Abousamra_2021_ICCV,
    author    = {Abousamra, Shahira and Belinsky, David and Van Arnam, John and Allard, Felicia and Yee, Eric and Gupta, Rajarsi and Kurc, Tahsin and Samaras, Dimitris and Saltz, Joel and Chen, Chao},  
    title     = {Multi-Class Cell Detection Using Spatial Context Representation},  
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},  
    year      = {2021},  
	}