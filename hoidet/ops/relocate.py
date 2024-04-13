"""
Relocate data

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch

from torch import Tensor
from typing import Optional, Union, List, Tuple, Dict, TypeVar

__all__ = ['relocate_to_cpu', 'relocate_to_cuda', 'relocate_to_device']

GenericTensor = TypeVar('GenericTensor', Tensor, List[Tensor], Tuple[Tensor, ...], Dict[str, Tensor])


def relocate_to_cpu(x: GenericTensor, ignore: bool = False) -> GenericTensor:
    """
    将Tensor、list、dict、tuple等类型的数据迁移到CPU上

    Args:
        x: 待迁移的数据
        ignore: 默认为False，表示当输入x是不支持的类型时，抛出异常；
                如果为True，则表示当输入x是不支持的类型时，忽略本次操作，do nothing

    Returns: 返回迁移到CPU上的数据

    """
    if isinstance(x, Tensor):
        return x.cpu()
    elif x is None:
        return x
    elif isinstance(x, list):
        return [relocate_to_cpu(item, ignore=ignore) for item in x]
    elif isinstance(x, tuple):
        return tuple(relocate_to_cpu(item, ignore=ignore) for item in x)
    elif isinstance(x, dict):
        for key in x:
            x[key] = relocate_to_cpu(x[key], ignore=ignore)
        return x
    elif not ignore:
        raise TypeError('Unsupported type of data {}'.format(type(x)))


def relocate_to_cuda(
        x: GenericTensor, ignore: bool = False,
        device: Optional[Union[torch.device, int]] = None,
        **kwargs
) -> GenericTensor:
    """
    将Tensor、list、dict、tuple等类型的数据迁移到GPU上
    
    Parameters:
    -----------
    x: Tensor, List[Tensor], Tuple[Tensor] or Dict[Tensor]
        待迁移的数据
    ignore: bool
        默认为False，表示当输入x是不支持的类型时，抛出异常；
        如果为True，则表示当输入x是不支持的类型时，忽略本次操作，do nothing
    device: torch.device or int
        GPU设备，可以是 torch.device 或者是 GPU 编号（从0开始）
    kwargs: dict
        用以 torch.Tensor.cuda() 的其他参数，很少使用

    Returns:
    --------
    Tensor, List[Tensor], Tuple[Tensor] or Dict[Tensor]
        Relocated tensor data
    """
    if isinstance(x, torch.Tensor):
        return x.cuda(device, **kwargs)
    elif x is None:
        return x
    elif isinstance(x, list):
        return [relocate_to_cuda(item, ignore, device, **kwargs) for item in x]
    elif isinstance(x, tuple):
        return tuple(relocate_to_cuda(item, ignore, device, **kwargs) for item in x)
    elif isinstance(x, dict):
        for key in x:
            x[key] = relocate_to_cuda(x[key], ignore, device, **kwargs)
        return x
    elif not ignore:
        raise TypeError('Unsupported type of data {}'.format(type(x)))


def relocate_to_device(
        x: GenericTensor, ignore: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        **kwargs
) -> GenericTensor:
    """
    将Tensor、list、dict、tuple等类型的数据迁移到指定设备上
    
    Parameters:
    -----------
    x: Tensor, List[Tensor], Tuple[Tensor] or Dict[Tensor]
        待迁移的数据
    ignore: bool
        默认为False，表示当输入x是不支持的类型时，抛出异常；
        如果为True，则表示当输入x是不支持的类型时，忽略本次操作，do nothing
    device: torch.device, str or int
        Destination device
    kwargs: dict
        用以 torch.Tensor.cuda() 的其他参数，很少使用

    Returns:
    --------
    Tensor, List[Tensor], Tuple[Tensor] or Dict[Tensor]
        Relocated tensor data
    """
    if isinstance(x, torch.Tensor):
        return x.to(device, **kwargs)
    elif x is None:
        return x
    elif isinstance(x, list):
        return [relocate_to_device(item, ignore, device, **kwargs) for item in x]
    elif isinstance(x, tuple):
        return tuple(relocate_to_device(item, ignore, device, **kwargs) for item in x)
    elif isinstance(x, dict):
        for key in x:
            x[key] = relocate_to_device(x[key], ignore, device, **kwargs)
        return x
    elif not ignore:
        raise TypeError('Unsupported type of data {}'.format(type(x)))
