#!/bin/bash

source activate mybase

nohup bash concmap_2_mask.sh &


wait;

exit 0
