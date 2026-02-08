# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-06,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#     type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
#     constructor='LayerDecayOptimizerConstructor_ViT', 
#     paramwise_cfg=dict(
#         num_layers=12, 
#         layer_decay_rate=0.9,
#         ),
#     grad_clip=dict(max_norm=1.0, norm_type=2)
#         )


optim_wrapper = dict(
    type='AmpOptimWrapper',  # ←←← 关键修改：启用混合精度！
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    constructor='LayerDecayOptimizerConstructor_ViT', 
    paramwise_cfg=dict(
        num_layers=12, 
        layer_decay_rate=0.9,
    )
)

clip_grad=dict(max_norm=1.0, norm_type=2)  # ←← 改成 clip_grad！