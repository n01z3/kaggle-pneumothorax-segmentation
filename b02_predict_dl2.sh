#!/usr/bin/env bash

MODEL=%1
echo ${1}
predict_fold() {
  echo ${1} ${2}
  CUDA_VISIBLE_DEVICES=${2} python n06_predict.py --fold=${1}
}
export -f predict_fold
parallel -j4 --line-buffer predict_fold {} '$(({%} % 4))' ::: {0..7}