dataset_type = "CocoDataset"
data_root = "/home/yuchunli/_DATASET/Vitens-Coliform-coco/"
backend_args = None

metainfo = {
    "classes": ("coliform"),
}

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type="CocoDataset",
        metainfo=metainfo,
        data_root=data_root,
        ann_file="annotations/instances_train.json",
        data_prefix=dict(img="images/train/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=None,
    ),
    pin_memory=True,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        metainfo=metainfo,
        data_root=data_root,
        ann_file="annotations/instances_val.json",
        data_prefix=dict(img="images/val/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        metainfo=metainfo,
        data_root=data_root,
        ann_file="annotations/instances_test.json",
        data_prefix=dict(img="images/test/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None,
    ),
)

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/instances_val.json",
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=None,
)

test_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/instances_test.json",
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=None,
)
