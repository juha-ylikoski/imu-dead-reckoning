from ronin.model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule


_input_channel, _output_channel = 6, 2
_fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}


def get_model():
    """Get ronin ResNet model

    Returns:
        nn.Module: PyTorch model
    """
    network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [2, 2, 2, 2],
                        base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    return network
