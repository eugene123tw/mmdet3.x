_base_ = ["./datasets/berry23.py"]

batch_augments = [
    dict(
        img_pad_value=0,
        mask_pad_value=0,
        pad_mask=True,
        pad_seg=False,
        size=(
            1024,
            1024,
        ),
        type="BatchFixedSizePad",
    ),
]

data_preprocessor = dict(
    batch_augments=batch_augments,
    bgr_to_rgb=True,
    mask_pad_value=0,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_mask=True,
    pad_seg=False,
    pad_size_divisor=32,
    seg_pad_value=255,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type="DetDataPreprocessor",
)

model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=-1,
        init_cfg=dict(checkpoint="torchvision://resnet50", type="Pretrained"),
        norm_cfg=dict(requires_grad=False, type="BN"),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style="pytorch",
        type="ResNet",
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mask_pad_value=0,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_seg=True,
        pad_size_divisor=1,
        seg_pad_value=255,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type="DetDataPreprocessor",
    ),
    init_cfg=None,
    panoptic_fusion_head=dict(
        init_cfg=None,
        loss_panoptic=None,
        num_stuff_classes=0,
        num_things_classes=1,
        type="MaskFormerFusionHead",
    ),
    panoptic_head=dict(
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        loss_cls=dict(
            class_weight=[1.0] * 1 + [0.1],
            loss_weight=1.0,
            reduction="mean",
            type="CrossEntropyLoss",
            use_sigmoid=False,
        ),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=1.0,
            naive_dice=True,
            reduction="mean",
            type="DiceLoss",
            use_sigmoid=True,
        ),
        loss_mask=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=20.0,
            reduction="mean",
            type="FocalLoss",
            use_sigmoid=True,
        ),
        num_queries=100,
        num_stuff_classes=0,
        num_things_classes=1,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type="ReLU"),
            encoder=dict(
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type="ReLU"),
                        embed_dims=256,
                        feedforward_channels=2048,
                        ffn_drop=0.1,
                        num_fcs=2,
                    ),
                    self_attn_cfg=dict(
                        batch_first=True, dropout=0.1, embed_dims=256, num_heads=8
                    ),
                ),
                num_layers=6,
            ),
            norm_cfg=dict(num_groups=32, type="GN"),
            positional_encoding=dict(normalize=True, num_feats=128),
            type="TransformerEncoderPixelDecoder",
        ),
        positional_encoding=dict(normalize=True, num_feats=128),
        transformer_decoder=dict(
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    batch_first=True, dropout=0.1, embed_dims=256, num_heads=8
                ),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.1,
                    num_fcs=2,
                ),
                self_attn_cfg=dict(
                    batch_first=True, dropout=0.1, embed_dims=256, num_heads=8
                ),
            ),
            num_layers=6,
            return_intermediate=True,
        ),
        type="MaskFormerHead",
    ),
    test_cfg=dict(
        filter_low_score=False,
        instance_on=False,
        iou_thr=0.8,
        max_per_image=100,
        object_mask_thr=0.8,
        panoptic_on=True,
        semantic_on=False,
    ),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type="ClassificationCost", weight=1.0),
                dict(binary_input=True, type="FocalLossCost", weight=20.0),
                dict(eps=1.0, pred_act=True, type="DiceCost", weight=1.0),
            ],
            type="HungarianAssigner",
        ),
        sampler=dict(type="MaskPseudoSampler"),
    ),
    type="MaskFormer",
)

auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)

default_scope = "mmdet"

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)

optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type="AdamW",
        weight_decay=0.0001,
    ),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(decay_mult=1.0, lr_mult=0.1),
            query_embed=dict(decay_mult=0.0, lr_mult=1.0),
        ),
        norm_decay_mult=0.0,
    ),
    type="OptimWrapper",
)

param_scheduler = dict(
    begin=0, by_epoch=True, end=12, gamma=0.1, milestones=[8, 11], type="MultiStepLR"
)

resume = False

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=15, val_interval=1)
test_cfg = dict(type="TestLoop")
val_cfg = dict(type="ValLoop")

vis_backends = [dict(type="LocalVisBackend")]

visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
    ],
)

launcher = "none"
load_from = "https://download.openmmlab.com/mmdetection/v3.0/maskformer/maskformer_r50_ms-16xb1-75e_coco/maskformer_r50_ms-16xb1-75e_coco_20230116_095226-baacd858.pth"
log_level = "INFO"
