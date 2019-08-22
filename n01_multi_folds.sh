#!/usr/bin/env bash

train_fold() {
  echo ${1} ${2}
  CUDA_VISIBLE_DEVICES=${2} python train_b.py --fold=${1} | tee ${1}+'fold.log'
}
export -f train_fold
parallel -j5 --line-buffer train_fold {} '$(({%} % 5 +1))' ::: {0..4}
