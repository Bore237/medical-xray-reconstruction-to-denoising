from monai.networks.nets import UNet
import torch

def load_ae_model(channels: tuple, stride: tuple, num_bloc_conv : int = 2):
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(
        spatial_dims = 2,
        in_channels = 1,
        out_channels = 1,
        channels = channels,
        strides = stride,
        num_res_units = num_bloc_conv,
        bias= True,
        act = 'PRELU', #'LeakyRELU'
        norm = 'BATCH', #INSTANCE' 'BATCH'  'GROUP'
        adn_ordering = 'NA', #'NDA" => bias = False
    )

    model = model.to(device)

    return model, device