#!/usr/bin/env bash
while true
do
    rsync -zavhP dgx:/raid/data_share/code/kaggle-pneumothorax-segmentation/se*log stage2/se154/
    rsync -zavhP dl2:/mnt/ssd1/n01z3/kaggle-pneumothorax-segmentation/*sx50*.log stage2/sx50p/
    rsync -zavhP dl4:/mnt/ssd2/n01z3/kaggle-pneumothorax-segmentation/*sx50*.log stage2/sx50
    sleep 1200
done