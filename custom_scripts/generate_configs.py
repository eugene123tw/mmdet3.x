import argparse
import copy
from pathlib import Path
from typing import List

from datumaro.components.annotation import AnnotationType
from datumaro.components.dataset import Dataset
from mmengine.config import Config


def generate_configs(
    config_path: Path,
    save_path: Path,
    work_dir: Path,
    epochs: int,
    batch_size: int,
    load_from: str,
    dataset_root: Path,
    dataset_list: List[str],
):
    assert config_path.exists(), "config_path does not exist"
    assert dataset_root.exists(), "dataset_root does not exist"
    save_path.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(config_path)
    cfg.load_from = load_from

    for dataset_name in dataset_list:
        _cfg = copy.deepcopy(cfg)

        dataset_path = dataset_root / dataset_name

        daturmaro_ds = Dataset.import_from(path=dataset_path, format="coco")
        dataset_labels = [
            label_item.name
            for label_item in daturmaro_ds.categories()
            .get(AnnotationType.label, None)
            .items
        ]
        metainfo = dict(
            classes=dataset_labels,
        )
        num_classes = len(dataset_labels)

        train_anno_path = dataset_path / "annotations/instances_train.json"
        val_anno_path = dataset_path / "annotations/instances_val.json"
        test_anno_path = dataset_path / "annotations/instances_test.json"
        train_image_prefix = "images/train"
        val_image_prefix = "images/val"
        test_image_prefix = "images/test"

        assert dataset_path.exists(), f"dataset_path does not exist: {dataset_path}"
        assert (
            train_anno_path.exists()
        ), f"train_anno_path does not exist: {train_anno_path}"
        assert val_anno_path.exists(), f"val_anno_path does not exist: {val_anno_path}"
        assert (
            test_anno_path.exists()
        ), f"test_anno_path does not exist: {test_anno_path}"
        assert (
            dataset_path / train_image_prefix
        ).exists(), (
            f"train_image_prefix does not exist: {dataset_path / train_image_prefix}"
        )
        assert (
            dataset_path / val_image_prefix
        ).exists(), (
            f"val_image_prefix does not exist: {dataset_path / val_image_prefix}"
        )

        _cfg.model.bbox_head.num_classes = num_classes
        _cfg.default_hooks.checkpoint.pop("max_keep_ckpts", None)
        _cfg.default_hooks.checkpoint.save_best = "auto"

        _cfg.custom_hooks.append(
            dict(
                type="EarlyStoppingHook",
                monitor="coco/segm_mAP",
                min_delta=0.01,
                patience=10,
            ),
        )

        _cfg.max_epochs = epochs
        _cfg.train_cfg.max_epochs = epochs
        _cfg.train_cfg.val_interval = 1
        _cfg.train_cfg.dynamic_intervals = [(epochs - 20, 1)]

        if _cfg.train_dataloader.batch_size != batch_size:
            print(
                f"Changing batch size from {_cfg.train_dataloader.batch_size} to {batch_size}"
            )
            _cfg.train_dataloader.batch_size = batch_size
        _cfg.train_dataloader.dataset.data_root = str(dataset_path)
        _cfg.train_dataloader.dataset.ann_file = str("annotations/instances_train.json")
        _cfg.train_dataloader.dataset.data_prefix.img = str(train_image_prefix)
        _cfg.train_dataloader.dataset.metainfo = metainfo

        # simplify dataset pipeline
        # _cfg.train_dataloader.dataset.pipeline = [
        #     dict(type='LoadImageFromFile', backend_args=None),
        #     dict(
        #         type='LoadAnnotations',
        #         with_bbox=True,
        #         with_mask=True,
        #         poly2mask=False),
        #     dict(type='Resize', scale=(640, 640), keep_ratio=True),
        #     dict(type='RandomFlip', prob=0.5),
        #     dict(
        #         type='Pad', size=(640, 640),
        #         pad_val=dict(img=(114, 114, 114))),
        #     dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
        #     dict(type='PackDetInputs')
        # ]

        # pop pipeline switch hook
        custom_hooks = []
        for hook in _cfg.custom_hooks:
            if hook["type"] == "PipelineSwitchHook":
                hook["switch_epoch"] = epochs - 20
            custom_hooks.append(hook)
        _cfg.custom_hooks = custom_hooks

        for key, hook in _cfg.default_hooks.items():
            if key == "checkpoint":
                hook["interval"] = 10
            if key == "logger":
                hook["interval"] = 10
            if key == "visualization":
                hook["interval"] = 10
                hook["draw"] = True

        #
        for scheduler in _cfg.param_scheduler:
            if scheduler.type == "CosineAnnealingLR":
                scheduler.begin = epochs // 2
                scheduler.end = epochs
                scheduler.T_max = epochs // 2

        # _cfg.val_dataloader.batch_size = 1
        _cfg.val_dataloader.dataset.data_root = str(dataset_path)
        _cfg.val_dataloader.dataset.ann_file = str("annotations/instances_val.json")
        _cfg.val_dataloader.dataset.data_prefix.img = str(val_image_prefix)
        _cfg.val_dataloader.dataset.metainfo = metainfo

        # _cfg.test_dataloader.batch_size = 1
        _cfg.test_dataloader.dataset.data_root = str(dataset_path)
        _cfg.test_dataloader.dataset.ann_file = str("annotations/instances_test.json")
        _cfg.test_dataloader.dataset.data_prefix.img = str(test_image_prefix)
        _cfg.test_dataloader.dataset.metainfo = metainfo

        _cfg.val_evaluator.ann_file = str(val_anno_path)
        _cfg.test_evaluator.ann_file = str(test_anno_path)

        _cfg.work_dir = str(work_dir / dataset_name)

        _cfg.dump(save_path / f"{dataset_name}.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--load_from", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, required=True)

    args = parser.parse_args()

    # config = "rtmdet/rtmdet-ins_s_8xb32-300e_coco.py"
    # folder_name = "rtmdet-ins_s"
    # load_from = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth"

    vitens_coliform = "Vitens-Coliform-coco"
    vitens_aeromonas = "Vitens-Aeromonas-coco"

    dataset_list = [vitens_coliform, vitens_aeromonas]

    config_path = Path(args.config)
    dataset_root = Path("/home/yuchunli/_DATASET/")
    save_path = Path(args.save_path)
    work_dir = Path(args.work_dir)

    generate_configs(
        config_path,
        save_path,
        work_dir,
        args.epochs,
        args.batch_size,
        args.load_from,
        dataset_root,
        dataset_list,
    )
