#!/bin/bash
WORK_DIR="work_dirs/rtmdet-ins_tiny"

AEROMONAS="Vitens-Aeromonas-coco"
COLIFORM="Vitens-Coliform-coco"
COLIFORM_24="Vitens-Coliform-coco-24"
KIEMGETAL="Vitens-Kiemgetal-coco-full"
DOTA="dota_v1_coco"
WGIDS="wgisd-coco"

ALUMUNIUM=alumunium-coco-roboflow
CHICKEN=Chicken-Real-Time-coco-roboflow
FASHION=fashion-categories-coco-roboflow
SKIN=skindetect-roboflow
POOL=pool-danger-coco-roboflow
BLUEBERRY=blueberries-roboflow

DATASET_ARRAY=(${COLIFORM} ${COLIFORM_24} ${AEROMONAS} ${KIEMGETAL} ${DOTA} ${WGIDS} ${ALUMUNIUM} ${CHICKEN} ${FASHION} ${SKIN} ${POOL} ${BLUEBERRY})

for dataset in ${DATASET_ARRAY[@]}; do
    CKPT="$(ls ${WORK_DIR}/$dataset | grep best)"
    python tools/test.py ${WORK_DIR}/$dataset/config.py ${WORK_DIR}/$dataset/${CKPT} --work-dir ${WORK_DIR}/$dataset/test
done
