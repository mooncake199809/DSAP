from .resnet_fpn import ResNetFPN_8_2


def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        if config['resolution'] == (8, 2):
            return ResNetFPN_8_2(config['resnetfpn'])
    else:
        raise ValueError(f"DSAP.BACKBONE_TYPE {config['backbone_type']} not supported.")
