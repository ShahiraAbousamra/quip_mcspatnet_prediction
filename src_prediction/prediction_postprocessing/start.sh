#!/bin/bash

source ../conf/variables.sh

nohup bash postprocessing.sh 0 4 &> ${LOG_OUTPUT_FOLDER}/log.postprocessing.thread_0.txt &
nohup bash postprocessing.sh 1 4 &> ${LOG_OUTPUT_FOLDER}/log.postprocessing.thread_1.txt &
nohup bash postprocessing.sh 2 4 &> ${LOG_OUTPUT_FOLDER}/log.postprocessing.thread_2.txt &
nohup bash postprocessing.sh 3 4 &> ${LOG_OUTPUT_FOLDER}/log.postprocessing.thread_3.txt &
wait

nohup bash call_run_poly_para_argmax.sh &> ${LOG_OUTPUT_FOLDER}/log.generatejson.txt &
exit 0
