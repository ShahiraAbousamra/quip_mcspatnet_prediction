#!/bin/bash

# Variables
DEFAULT_OBJ=40
DEFAULT_MPP=0.25
CANCER_TYPE=quip
MONGODB_HOST=osprey.bmi.stonybrook.edu
MONGODB_PORT=27017
HEATMAP_VERSION=MCSpatNet-BRCA
if [[ ! -n $LYM_PREDICTION_BATCH_SIZE ]]; then
   LYM_PREDICTION_BATCH_SIZE=8;
fi
# Base directory
BASE_DIR=/home/sabousamra/quip_mcspatnet_prediction/src_prediction/
OUT_DIR=/data03/shared/sabousamra/brca_mcspatnet/output/
POSTPROCESS_DIR=${OUT_DIR}/wsi_patches_postprocessing/
PREDICTION_INTERMEDIATE_FOLDER=${OUT_DIR}/wsi_patches_intermediate/

# Paths of data, log, input, and output
JSON_OUTPUT_FOLDER=${OUT_DIR}/json_quip
BINARY_JSON_OUTPUT_FOLDER=${OUT_DIR}/heatmap_jsons_binary
HEATMAP_TXT_OUTPUT_FOLDER=${OUT_DIR}/heatmap_txt
BINARY_HEATMAP_TXT_OUTPUT_FOLDER=${OUT_DIR}/heatmap_txt_binary
LOG_OUTPUT_FOLDER=${OUT_DIR}/log
CSV_OUTPUT_FOLDER=${OUT_DIR}/csv_point_coord

SVS_INPUT_PATH=/data03/shared/sabousamra/TCGA_BRCA_DX1/
PATCH_PATH=/data03/shared/sabousamra/TCGA_BRCA_DX1_patches/


LYM_NECRO_CNN_MODEL_PATH=/home/sabousamra/quip_mcspatnet_prediction/src_prediction/prediction/mcspatnet_models/mcspat_brca-m2c.pth
EXTERNAL_LYM_MODEL=1

# create missing output directories
if [ ! -d ${OUT_DIR} ]; then
  mkdir ${OUT_DIR} ;
fi

if [ ! -d ${JSON_OUTPUT_FOLDER} ]; then
  mkdir ${JSON_OUTPUT_FOLDER} ;
fi


if [ ! -d ${LOG_OUTPUT_FOLDER} ]; then
  mkdir ${LOG_OUTPUT_FOLDER} ;
fi

if [ ! -d ${PREDICTION_INTERMEDIATE_FOLDER} ]; then
  mkdir ${PREDICTION_INTERMEDIATE_FOLDER} ;
fi

if [ ! -d ${POSTPROCESS_DIR} ]; then
  mkdir ${POSTPROCESS_DIR} ;
fi

if [ ! -d ${CSV_OUTPUT_FOLDER} ]; then
  mkdir ${CSV_OUTPUT_FOLDER} ;
fi
