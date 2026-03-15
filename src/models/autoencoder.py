from monai.networks.nets import UNet

def load_model(channels: tuple, stride: tuple, num_bloc_conv : int = 2):

    model = UNet(
        spatial_dims = 2,
        in_channels = 1,
        out_channels = 1,
        channels = channels,
        strides = stride,
        num_res_units = num_bloc_conv,
        bias= True,
        act = 'PRELU', #'LeakyRELU'
        norm = 'INSTANCE', #'BATCH'  'GROUP'
        adn_ordering = 'NAD', #'NDA" => bias = False
    )

    return model