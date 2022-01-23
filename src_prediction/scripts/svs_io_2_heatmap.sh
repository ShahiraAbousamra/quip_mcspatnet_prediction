#!/bin/bash

if [[ -n $BASE_DIR ]]; then
	cd $BASE_DIR
else
	cd ../
fi

source ./conf/variables.sh

cd patch_extraction
nohup bash start_io.sh &
cd ..

cd prediction
nohup bash start_seg.sh  &
cd ..

wait;

cd prediction_postprocessing
nohup bash start.sh &
cd ..


wait;

wait;

exit 0
