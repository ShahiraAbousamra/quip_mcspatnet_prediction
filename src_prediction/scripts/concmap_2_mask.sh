#!/bin/bash

if [[ -n $BASE_DIR ]]; then
	cd $BASE_DIR
else
	cd ../
fi

source ./conf/variables.sh

cd prediction_postprocessing
nohup bash start.sh &
cd ..


wait;



wait;

exit 0
