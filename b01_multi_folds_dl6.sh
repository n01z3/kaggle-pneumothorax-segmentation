#!/usr/bin/env bash

MODEL=%1

train_fold() {
  echo ${1} ${2}
  CUDA_VISIBLE_DEVICES=${2} python n15_train.py --fold=${1} --net $MODEL | tee ${1}'fold_1.log'
}
export -f train_fold
parallel -j4 --line-buffer train_fold {} '$(({%} % 4))' ::: {0..7}
