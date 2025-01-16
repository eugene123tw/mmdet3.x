_base_ = "../retinanet/retinanet_x101-32x4d_fpn_1x_coco.py"

model = dict(
    bbox_head=dict(
        type="PISARetinaHead",
        loss_bbox=dict(type="SmoothL1Loss", beta=0.11, loss_weight=1.0),
    ),
    train_cfg=dict(isr=dict(k=2.0, bias=0.0), carl=dict(k=1.0, bias=0.2)),
)
