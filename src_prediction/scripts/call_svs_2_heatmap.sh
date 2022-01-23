#!/bin/bash

source activate mybase

nohup bash svs_2_heatmap.sh &


wait;

exit 0
