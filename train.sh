#!/usr/bin/env bash


train_fold() {
  echo ${1} ${2}
  CUDA_VISIBLE_DEVICES=${2} python n15_train.py --fold=${1} --net 'se154' | tee 'se154'${1}'fold_2.log'
}
export -f train_fold
parallel -j8 --line-buffer train_fold {} '$(({%} % 8))' ::: {0..7}


train_fold() {
  echo ${1} ${2}
  CUDA_VISIBLE_DEVICES=${2} python n15_train.py --fold=${1} --net 'sx101' | tee 'sx101'${1}'fold_2.log'
}
export -f train_fold
parallel -j8 --line-buffer train_fold {} '$(({%} % 8))' ::: {0..7}



train_fold() {
  echo ${1} ${2}
  CUDA_VISIBLE_DEVICES=${2} python n15_train.py --fold=${1} --net 'sx50' | tee 'sx50'${1}'fold_2.log'
}
export -f train_fold
parallel -j8 --line-buffer train_fold {} '$(({%} % 8))' ::: {0..7}



train_fold() {
  echo ${1} ${2}
  CUDA_VISIBLE_DEVICES=${2} python n15_train.py --fold=${1} --net 'sxh101' | tee 'sxh101'${1}'fold_2.log'
}
export -f train_fold
parallel -j8 --line-buffer train_fold {} '$(({%} % 8))' ::: {0..7}



train_fold() {
  echo ${1} ${2}
  CUDA_VISIBLE_DEVICES=${2} python n15_train.py --fold=${1} --net 'sxh50' | tee 'sxh50'${1}'fold_2.log'
}
export -f train_fold
parallel -j8 --line-buffer train_fold {} '$(({%} % 8))' ::: {0..7}


