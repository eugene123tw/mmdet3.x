dataset_type = 'CocoDataset'
data_root = '/home/yuchunli/_DATASET/mosquito_data/'
backend_args = None

metainfo = {
    'classes': (
        "aegypti",
        "albopictus",
        "anopheles",
        "culex",
        "culiseta",
        "japonicus/koreicus"
    ),
}

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='anno/fold_0/annotations/instances_train.json',
        data_prefix=dict(img='train_images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=None),
    pin_memory=True)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='anno/fold_0/annotations/instances_val.json',
        data_prefix=dict(img='train_images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'anno/fold_0/annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    backend_args=None,
    proposal_nums=(100, 1, 10))

test_evaluator = val_evaluator

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))

img_scales = [(640, 640), (320, 320), (960, 960)]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                {
                    'type': 'Resize',
                    'scale': (640, 640),
                    'keep_ratio': True
                },
                {
                    'type': 'Resize',
                    'scale': (320, 320),
                    'keep_ratio': True
                }, {
                    'type': 'Resize',
                    'scale': (960, 960),
                    'keep_ratio': True
                }
            ],
            [
                {
                    'type': 'RandomFlip',
                    'prob': 1.0
                },
                {
                    'type': 'RandomFlip',
                    'prob': 0.0
                }
            ],
            [
                {
                    'type': 'Pad',
                    'size': (960, 960),
                    'pad_val': {'img': (114, 114, 114)}
                }
            ],
            [
                {
                    'type': 'LoadAnnotations', 'with_bbox': True
                }
            ],
            [
                {
                    'type':
                    'PackDetInputs',
                    'meta_keys':
                    (
                        'img_id', 'img_path', 'ori_shape',
                        'img_shape', 'scale_factor',
                        'flip', 'flip_direction'
                    )
                }
            ]
        ]
    )
]
