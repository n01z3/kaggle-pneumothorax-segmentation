#!/usr/bin/env bash


bash b02_predict_dl2.sh

python n16_submit.py --model_name sx101 --correction True
python n16_submit.py --model_name sx101 --correction False
python n16_submit.py --model_name sx50 --correction True
python n16_submit.py --model_name sx50 --correction False
python n16_submit.py --model_name se154 --correction True
python n16_submit.py --model_name se154 --correction False
python n16_submit.py --model_name sxh101 --correction True
python n16_submit.py --model_name sxh101 --correction False
python n16_submit.py --model_name sxh50 --correction True
python n16_submit.py --model_name sxh50 --correction False

python n12_union_blend.py
python n12_union_blend_v2.py