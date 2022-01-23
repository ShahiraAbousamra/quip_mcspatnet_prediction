#!/bin/bash

source ../conf/variables.sh

COD_PARA=$1
MAX_PARA=$2
IN_FOLDER=${PREDICTION_INTERMEDIATE_FOLDER}
OUT_FOLDER=${POSTPROCESS_DIR}
OUT_FOLDER_CSV=${CSV_OUTPUT_FOLDER}
READY_FILE=patch-level-lym.txt
DONE_FILE=postprocessing_done.txt

PRE_FILE_NUM=0
while [ 1 ]; do

    LINE_N=0
    FILE_NUM=0
    EXTRACTING=0
	for files in ${IN_FOLDER}/*.*; do
        FILE_NUM=$((FILE_NUM+1))
        if [ ! -f ${files}/${READY_FILE} ]; then EXTRACTING=1; fi

		LINE_N=$((LINE_N+1))
		if [ $((LINE_N % MAX_PARA)) -ne ${COD_PARA} ]; then continue; fi

        if [ -f ${files}/${READY_FILE} ]; then
			SVS=`echo ${files} | awk -F'/' '{print $(NF)}'`			
			OUT_SVS_PATH=${OUT_FOLDER}/${SVS}
            if [ ! -f ${OUT_SVS_PATH}/${DONE_FILE} ]; then
                echo ${OUT_SVS_PATH}/${DONE_FILE} generating
				# python postprocessing.py $SVS $IN_FOLDER $OUT_FOLDER
				python postprocessing_csv.py $SVS $IN_FOLDER $OUT_FOLDER $OUT_FOLDER_CSV
				if [ $? -ne 0 ]; then
					echo "failed postprocessing patches for " ${SVS}
					rm -rf ${OUT_SVS_PATH}
				fi
            fi
        fi

	done

    if [ ${EXTRACTING} -eq 0 ] && [ ${PRE_FILE_NUM} -eq ${FILE_NUM} ]; then break; fi
    PRE_FILE_NUM=${FILE_NUM}
done
exit 0;

