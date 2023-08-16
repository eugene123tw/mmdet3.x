from typing import List
from pathlib import Path
from mmengine.config import Config


def generate_configs(
        batch_size: int,
        config_path: Path,
        output_root: Path,
        load_from: str,
        dataset_root: Path,
        dataset_list: List[str]):
    assert config_path.exists(), "config_path does not exist"
    assert dataset_root.exists(), "dataset_root does not exist"
    output_root.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(config_path)
    cfg.load_from = load_from

    for dataset in dataset_list:
        dataset_name = dataset["name"]
        dataset_labels = dataset["labels"]
        metainfo = dict(classes=dataset_labels,)
        num_classes = len(dataset_labels)
        output_path = output_root / dataset_name / ""
        dataset_path = dataset_root / dataset_name
        train_anno_path = dataset_path / "annotations/instances_train.json"
        val_anno_path = dataset_path / "annotations/instances_val.json"
        test_anno_path = dataset_path / "annotations/instances_test.json"
        train_image_prefix = "images/train"
        val_image_prefix = "images/val"
        test_image_prefix = "images/test"

        output_path.mkdir(parents=True, exist_ok=True)
        assert dataset_path.exists(), f"dataset_path does not exist: {dataset_path}"
        assert train_anno_path.exists(), f"train_anno_path does not exist: {train_anno_path}"
        assert val_anno_path.exists(), f"val_anno_path does not exist: {val_anno_path}"
        assert test_anno_path.exists(), f"test_anno_path does not exist: {test_anno_path}"
        assert (dataset_path / train_image_prefix).exists(), f"train_image_prefix does not exist: {dataset_path / train_image_prefix}"
        assert (dataset_path / val_image_prefix).exists(), f"val_image_prefix does not exist: {dataset_path / val_image_prefix}"

        cfg.model.bbox_head.num_classes = num_classes

        cfg.default_hooks.checkpoint.pop('max_keep_ckpts', None)
        cfg.default_hooks.checkpoint.save_best = 'auto'

        cfg.custom_hooks.append(
            dict(
                type='EarlyStoppingHook',
                monitor='coco/segm_mAP',
                min_delta=0.01,
                patience=10),
        )

        cfg.train_dataloader.batch_size = batch_size
        cfg.train_dataloader.dataset.data_root = str(dataset_path)
        cfg.train_dataloader.dataset.ann_file = str("annotations/instances_train.json")
        cfg.train_dataloader.dataset.data_prefix.img = str(train_image_prefix)
        cfg.train_dataloader.dataset.metainfo = metainfo

        cfg.val_dataloader.batch_size = 1
        cfg.val_dataloader.dataset.data_root = str(dataset_path)
        cfg.val_dataloader.dataset.ann_file = str("annotations/instances_val.json")
        cfg.val_dataloader.dataset.data_prefix.img = str(val_image_prefix)
        cfg.val_dataloader.dataset.metainfo = metainfo

        cfg.test_dataloader.batch_size = 1
        cfg.test_dataloader.dataset.data_root = str(dataset_path)
        cfg.test_dataloader.dataset.ann_file = str("annotations/instances_test.json")
        cfg.test_dataloader.dataset.data_prefix.img = str(test_image_prefix)
        cfg.test_dataloader.dataset.metainfo = metainfo

        cfg.val_evaluator.ann_file = str(val_anno_path)
        cfg.test_evaluator.ann_file = str(test_anno_path)

        cfg.work_dir = str(output_path)

        cfg.dump(output_path / "config.py")


if __name__ == '__main__':
    config = "rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py"
    folder_name = "rtmdet-ins_tiny"
    load_from = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth"

    # config = "rtmdet/rtmdet-ins_s_8xb32-300e_coco.py"
    # folder_name = "rtmdet-ins_s"
    # load_from = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth"

    batch_size = 16
    vitens_coliform = dict(name="Vitens-Coliform-coco", labels=["coliform"])
    vitens_aeromonas = dict(name="Vitens-Aeromonas-coco", labels=["aeromonas"])

    dataset_list = [vitens_coliform, vitens_aeromonas]

    config_root = Path('/home/yuchunli/git/mmdet_3x/configs/')
    dataset_root = Path('/home/yuchunli/_DATASET/')
    output_root = Path('/home/yuchunli/git/mmdet_3x/work_dirs/')

    config_path = config_root / config
    output_path = output_root / folder_name

    generate_configs(
        batch_size,
        config_path,
        output_path,
        load_from,
        dataset_root,
        dataset_list)
