#!/usr/bin/env bash

train_fold() {
  echo ${1} ${2}
  CUDA_VISIBLE_DEVICES=${2} python train_orig.py --fold=${1} | tee ${1}'fold_1.log'
}
export -f train_fold
parallel -j4 --line-buffer train_fold {} '$(({%} % 4))' ::: {6..9}