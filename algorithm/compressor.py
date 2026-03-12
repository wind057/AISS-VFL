import torch
import numpy as np

# Paper title: Compressed-VFL: Communication-Efficient Learning with Vertically Partitioned Data
# Code repo: https://github.com/timcast725/C-VFL

def quantize_scalar(x, quant_min=0, quant_max=1, quant_level=5):
    """Uniform quantization approach

    Notebook: C2S2_DigitalSignalQuantization.ipynb

    Args:
        x: Original signal
        quant_min: Minimum quantization level
        quant_max: Maximum quantization level
        quant_level: Number of quantization levels

    Returns:
        x_quant: Quantized signal
    """
    device = x.device
    x_np = x.cpu().detach().numpy()
    
    # Move into 0,1 range:
    x_normalize = x_np/np.max(x_np)
    x_normalize = np.nan_to_num(x_normalize)

    dither = np.random.uniform(-(quant_max-quant_min)/(2*(quant_level-1)),
				(quant_max-quant_min)/(2*(quant_level-1)),
				size=x_normalize.shape)
    x_normalize = x_normalize + dither

    x_normalize = (x_normalize-quant_min) * (quant_level-1) / (quant_max-quant_min)
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = np.around(x_normalize)
    x_quant = (x_normalize_quant) * (quant_max-quant_min) / (quant_level-1) + quant_min

    # Move out of 0,1 range:
    x_quant = np.max(x_np)*(x_quant - dither)
    return torch.from_numpy(x_quant).float().to(device)

def topk(tensor, compress_ratio):
    """
    Get topk elements in tensor
    """
    shape = tensor.shape
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))
    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    numel = tensor.numel()
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed.view(shape)

