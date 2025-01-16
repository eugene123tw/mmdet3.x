default_scope = "mmdet"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=10, max_keep_ckpts=3),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer",
    vis_backends=[dict(type="LocalVisBackend")],
    name="visualizer",
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth"
resume = False
train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=20,
    val_interval=1,
    dynamic_intervals=[(280, 1)],
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-05, by_epoch=False, begin=0, end=1000),
    dict(
        type="CosineAnnealingLR",
        eta_min=0.0002,
        begin=150,
        end=300,
        T_max=150,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)
auto_scale_lr = dict(enable=False, base_batch_size=16)
dataset_type = "CocoDataset"
data_root = "/home/yuchunli/_DATASET/BlueBerry23.v1i.coco-mmdetection/"

metainfo = {
    "classes": ("berry"),
}
backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type="CachedMosaic",
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False,
    ),
    dict(
        type="RandomResize", scale=(1280, 1280), ratio_range=(0.5, 2.0), keep_ratio=True
    ),
    dict(type="RandomCrop", crop_size=(640, 640)),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type="CachedMixUp",
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5,
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1)),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=(640, 640), keep_ratio=True),
    dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        metainfo=metainfo,
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
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/instances_val.json",
        data_prefix=dict(img="images/val/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None,
    ),
)

test_dataloader = dict(
    batch_size=5,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        metainfo=metainfo,
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
    proposal_nums=(100, 1, 10),
)

test_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/instances_test.json",
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=None,
    proposal_nums=(100, 1, 10),
)

img_scales = [(640, 640), (320, 320), (960, 960)]

model = dict(
    type="RTMDet",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None,
    ),
    backbone=dict(
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        channel_attention=True,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU", inplace=True),
        init_cfg=dict(
            type="Pretrained",
            prefix="backbone.",
            checkpoint="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth",
        ),
    ),
    neck=dict(
        type="CSPNeXtPAFPN",
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU", inplace=True),
    ),
    bbox_head=dict(
        type="RTMDetInsSepBNHead",
        num_classes=1,
        in_channels=96,
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        feat_channels=96,
        act_cfg=dict(type="SiLU", inplace=True),
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        anchor_generator=dict(type="MlvlPointGenerator", offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type="DistancePointBBoxCoder"),
        loss_cls=dict(
            type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0
        ),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
        loss_mask=dict(type="DiceLoss", loss_weight=2.0, eps=5e-06, reduction="mean"),
    ),
    train_cfg=dict(
        assigner=dict(type="DynamicSoftLabelAssigner", topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5,
    ),
)
train_pipeline_stage2 = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type="RandomResize", scale=(640, 640), ratio_range=(0.5, 2.0), keep_ratio=True
    ),
    dict(
        type="RandomCrop",
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True,
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1)),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type="PackDetInputs"),
]
max_epochs = 300
stage2_num_epochs = 20
base_lr = 0.004
interval = 10
custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type="PipelineSwitchHook",
        switch_epoch=280,
        switch_pipeline=[
            dict(type="LoadImageFromFile", backend_args=None),
            dict(
                type="LoadAnnotations", with_bbox=True, with_mask=True, poly2mask=False
            ),
            dict(
                type="RandomResize",
                scale=(640, 640),
                ratio_range=(0.5, 2.0),
                keep_ratio=True,
            ),
            dict(
                type="RandomCrop",
                crop_size=(640, 640),
                recompute_bbox=True,
                allow_negative_crop=True,
            ),
            dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1)),
            dict(type="YOLOXHSVRandomAug"),
            dict(type="RandomFlip", prob=0.5),
            dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
            dict(type="PackDetInputs"),
        ],
    ),
]
checkpoint = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth"
launcher = "none"
work_dir = "./work_dirs/rtmdet-ins_tiny_berry"