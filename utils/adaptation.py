import torch
from torchvision.transforms import functional as T


def adapt_input(
    input,
    size,
    source_modal,
    target_modal,
    encoder_type="spectral_gpt",
    device=torch.device("cuda"),
):
    
    def adapt_input_tensor(
        tensor,
        size,
        source_bands,
        target_bands,
        encoder_type="spectral_gpt",
        device=torch.device("cuda"),
    ):   
              
        if len(tensor.shape) == 4:
            Bs, C, H, W = tensor.shape
            n_tensor = T.resize(
                img=tensor,
                size=(size, size),
                interpolation=T.InterpolationMode.BILINEAR,
            ).float()
            if encoder_type in ("prithvi"):
                n_tensor = n_tensor.unsqueeze(dim=2)
                Te = 1
        elif len(tensor.shape) == 5:
            Bs, C, Te, H, W = tensor.shape
            n_tensor = torch.empty((Bs, C, Te, size, size)).to(device).float()

            for i in range(Te):
                n_tensor[:, :, i, :, :] = T.resize(
                    img=tensor[:, :, i, :, :],
                    size=(size, size),
                    interpolation=T.InterpolationMode.BILINEAR,
                )
            if encoder_type in ("spectral_gpt"):
                n_tensor = n_tensor.squeeze()

        # Adapt from an arbitrary list of source bands to an arbitrary list of target bands
        # by moving the matching parts to the right place, and filling out the rest with zeros.
        if len(n_tensor.shape) == 4:
            zero_tensor = torch.zeros((Bs, 1, size, size)).to(device)
        elif len(n_tensor.shape) == 5:
            zero_tensor = torch.zeros((Bs, 1, Te, size, size)).to(
                device
            )

        source_band_indexes = [source_bands.index(t) if t in source_bands else None for t in target_bands]    
        out_tensors = [n_tensor[:, [i], ...] if i is not None else zero_tensor for i in source_band_indexes]
            
        return torch.concat(out_tensors, dim=1).to(device)
    
    # TODO: to support croma and dofa multi-modality

    tensor = input['optical'].to(device)
    source_bands = source_modal['optical']
    target_bands = target_modal['optical']
    return adapt_input_tensor(tensor, size, source_bands, target_bands, encoder_type, device)

    '''
    if encoder_type not in ["dofa", "croma"]:
        tensor = input['s2'].to(device)
        source_bands = source_modal['s2']
        target_bands = target_modal['s2']

        return adapt_input_tensor(tensor, size, source_bands, target_bands, encoder_type, device)
    else:
        output = []
        for modal in ["s1", "s2"]:
            # TODO: to support croma and dofa multi-modality
            if modal not in input:
                continue
            tensor = input[modal].to(device)
            source_bands = source_modal[modal]
            target_bands = target_modal[modal]
            input[modal] = adapt_input_tensor(tensor, size, source_bands, target_bands, encoder_type, device)
            output.append(input[modal])
        return output
    '''


def adapt_target(tensor, size, device=torch.device("cuda")):
    tensor = tensor.to(device)
    return T.resize(
        img=tensor, size=(size, size), interpolation=T.InterpolationMode.NEAREST
    ).squeeze(dim=1).long()
