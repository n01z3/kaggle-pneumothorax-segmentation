#!/usr/bin/env bash

train_fold() {
  echo ${1} ${2}
  CUDA_VISIBLE_DEVICES=${2} python n15_train.py --fold=${1} --net sx50 | tee ${1}'sx50_st2p_fold_1.log'
}
export -f train_fold
parallel -j4 --line-buffer train_fold {} '$(({%} % 4))' ::: {0..7}
