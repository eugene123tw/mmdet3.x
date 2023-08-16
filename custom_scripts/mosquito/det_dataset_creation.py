import glob
import json
import os
import warnings
import copy
import cv2
import mmcv
from pathlib import Path
import numpy as np
import pandas as pd
from datumaro import DatasetItem, Bbox
from datumaro.components.project import Dataset
from sklearn.model_selection import KFold


class MosquitoAlertDataset:

    def __init__(self, data_root) -> None:
        self.data_root = data_root
        self.labels = ["aegypti", "albopictus", "anopheles", "culex", "culiseta", "japonicus/koreicus"]
        self.df = pd.read_csv(os.path.join(self.data_root, 'train.csv'))
        self.dsitems = self._make_dsitems()

    def _make_dsitems(self):
        dsitems = []
        for index, row in self.df.iterrows():
            image_id = Path(row['img_fName'])
            attributes = {'filename': f'{self.data_root}/train_images/{image_id}'}       

            label_idx = self.labels.index(row['class_label'])
            x = row['bbx_xtl']
            y = row['bbx_ytl']
            w = row['bbx_xbr'] - row['bbx_xtl']
            h = row['bbx_ybr'] - row['bbx_ytl']

            bbox = Bbox(
                x=x,
                y=y,
                w=w,
                h=h,
                label=label_idx,
                attributes=attributes)

            dsitems.append(
                DatasetItem(
                    id=str(image_id.stem),
                    image=attributes['filename'],
                    annotations=[bbox],
                    attributes=attributes)
            )
        return dsitems

    def strategy_0(self, n_folds=5, seed=0, export_path=None):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (train_indices, val_indices) in enumerate(kf.split(range(len(self.dsitems)))):            
            for i in train_indices:
                self.dsitems[i].subset = 'train'
            for i in val_indices:
                self.dsitems[i].subset = 'val'
            if export_path is not None:
                dataset = Dataset.from_iterable(
                    self.dsitems, categories=self.labels)
                dataset.export(
                    f'{export_path}/fold_{fold}',
                    'coco',
                    default_image_ext='.jpeg')

    def export(self, dsitems, export_path, save_media=False):
        dataset = Dataset.from_iterable(dsitems, categories=self.labels)
        dataset.export(
            f'{export_path}',
            'coco',
            default_image_ext='.jpeg',
            save_media=save_media)

    def _get_unannotated_images(self):
        image_list = []
        for index, row in self.df.iterrows():
            if row['dataset'] == 3:
                image_list.append(row)
        return image_list


if __name__ == '__main__':
    dataset = MosquitoAlertDataset(
        data_root='/home/yuchunli/_DATASET/mosquito_data'
    )

    dataset.strategy_0(
        export_path='/home/yuchunli/_DATASET/mosquito_data/anno',
        n_folds=5,
        seed=0,
    )