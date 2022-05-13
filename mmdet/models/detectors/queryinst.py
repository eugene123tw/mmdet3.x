# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from .sparse_rcnn import SparseRCNN


@MODELS.register_module()
class QueryInst(SparseRCNN):
    r"""Implementation of
    `Instances as Queries <http://arxiv.org/abs/2105.01928>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(QueryInst, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)