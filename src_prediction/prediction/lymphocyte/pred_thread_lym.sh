#!/bin/bash

source ../../conf/variables.sh

OUT_FOLDER=${PREDICTION_INTERMEDIATE_FOLDER}
FOLDER=$1
# PARAL = [0, MAX_PARAL-1]
PARAL=$2
MAX_PARAL=$3
DEVICE=$4

DATA_FILE=patch-level-lym.txt
DONE_FILE=extraction_done.txt

if [ ${EXTERNAL_LYM_MODEL} -eq 0 ]; then
    EXEC_FILE=pred.py
else
    EXEC_FILE=pred_by_external_model.py
fi

PRE_FILE_NUM=0
while [ 1 ]; do
    LINE_N=0
    FILE_NUM=0
    EXTRACTING=0
    for files in ${FOLDER}/*/; do
        FILE_NUM=$((FILE_NUM+1))
        if [ ! -f ${files}/${DONE_FILE} ]; then EXTRACTING=1; fi

        LINE_N=$((LINE_N+1))
        if [ $((LINE_N % MAX_PARAL)) -ne ${PARAL} ]; then continue; fi

		SVS=`echo ${files} | awk -F'/' '{print $(NF-1)}'`
		OUT_SVS_PATH=${OUT_FOLDER}/${SVS}
        if [ -f ${files}/${DONE_FILE} ]; then
            if [ ! -f ${OUT_SVS_PATH}/${DATA_FILE} ]; then
                echo ${OUT_SVS_PATH}/${DATA_FILE} generating
                THEANO_FLAGS="device=${DEVICE}" python -u ${EXEC_FILE} \
                    ${files} ${LYM_NECRO_CNN_MODEL_PATH} ${DATA_FILE} ${LYM_PREDICTION_BATCH_SIZE} ${OUT_SVS_PATH}
            fi
        fi
    done

    if [ ${EXTRACTING} -eq 0 ] && [ ${PRE_FILE_NUM} -eq ${FILE_NUM} ]; then break; fi
    PRE_FILE_NUM=${FILE_NUM}
done

exit 0
