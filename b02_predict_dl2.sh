#!/usr/bin/env bash

train_fold() {
  echo ${1} ${2}
  CUDA_VISIBLE_DEVICES=${2} python n06_predict.py --fold=${1}
}
export -f train_fold
parallel -j4 --line-buffer train_fold {} '$(({%} % 4))' ::: {0..7}
