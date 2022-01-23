#!/bin/bash

#source activate shahira_env
source ../conf/variables.sh


IN_FOLDER=${POSTPROCESS_DIR}
OUT_FOLDER=${JSON_OUTPUT_FOLDER}
EXEC_FILE1=./generating_polygons_and_meta_files_for_quip/1_run_poly_para_argmax.py
EXEC_FILE2=./generating_polygons_and_meta_files_for_quip/2_run_json.py
READY_FILE=postprocessing_done.txt
DONE_FILE1=json_done.txt
DONE_FILE2=poly_done.txt

# for files in ${IN_FOLDER}/*/; do
	# echo ${files}
	# python -u ${EXEC_FILE1} \
				# ${IN_FOLDER} ${OUT_FOLDER} ${files}	
# done

# for files in ${IN_FOLDER}/*/; do
	# echo ${files}
	# python -u ${EXEC_FILE2} \
				# ${IN_FOLDER} ${OUT_FOLDER} ${files} ${SVS_INPUT_PATH}
# done


PRE_FILE_NUM=0
while [ 1 ]; do

    # LINE_N=0
    FILE_NUM=0
    EXTRACTING=0
	for files in ${IN_FOLDER}/*.*; do
        FILE_NUM=$((FILE_NUM+1))
        if [ ! -f ${files}/${READY_FILE} ]; then EXTRACTING=1; fi

		# LINE_N=$((LINE_N+1))
		# if [ $((LINE_N % MAX_PARA)) -ne ${COD_PARA} ]; then continue; fi

        if [ -f ${files}/${READY_FILE} ]; then
			SVS=`echo ${files} | awk -F'/' '{print $NF}'`
			OUT_SVS_PATH=${OUT_FOLDER}/${SVS}
            if [ ! -f ${OUT_SVS_PATH}/${DONE_FILE1} ]; then
                echo ${OUT_SVS_PATH}/${DONE_FILE1} generating
				python -u ${EXEC_FILE1} \
							${IN_FOLDER} ${OUT_FOLDER} ${files}	
				touch ${OUT_SVS_PATH}/${DONE_FILE1}
			fi
            if [ ! -f ${OUT_SVS_PATH}/${DONE_FILE2} ]; then
                echo ${OUT_SVS_PATH}/${DONE_FILE2} generating
				python -u ${EXEC_FILE2} \
							${IN_FOLDER} ${OUT_FOLDER} ${files} ${SVS_INPUT_PATH} ${HEATMAP_VERSION}
				touch ${OUT_SVS_PATH}/${DONE_FILE2}
            fi
        fi

	done

    if [ ${EXTRACTING} -eq 0 ] && [ ${PRE_FILE_NUM} -eq ${FILE_NUM} ]; then break; fi
    PRE_FILE_NUM=${FILE_NUM}
done
exit 0;
