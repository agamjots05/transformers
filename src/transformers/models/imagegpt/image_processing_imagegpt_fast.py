# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fast Image processor class for ImageGPT."""

import torch
from typing import Optional, Union

from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import PILImageResampling
from ...utils import auto_docstring


def squared_euclidean_distance_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute squared Euclidean distances between all pixels and clusters.
    
    Args:
        a: (N, 3) tensor of pixel RGB values
        b: (M, 3) tensor of cluster RGB values
        
    Returns:
        (N, M) tensor of squared distances
    """
    b = b.t()  # (3, M)
    a2 = torch.sum(a ** 2, dim=1)  # (N,)
    b2 = torch.sum(b ** 2, dim=0)  # (M,)
    ab = torch.matmul(a, b)        # (N, M)
    d = a2[:, None] - 2 * ab + b2[None, :]
    return d


def color_quantize_torch(x: torch.Tensor, clusters: torch.Tensor) -> torch.Tensor:
    """
    Assign each pixel to its nearest color cluster.
    
    Args:
        x: (H*W, 3) tensor of flattened pixel RGB values
        clusters: (n_clusters, 3) tensor of cluster RGB values
        
    Returns:
        (H*W,) tensor of cluster indices
    """
    d = squared_euclidean_distance_torch(x, clusters)
    return torch.argmin(d, dim=1)


@auto_docstring
class ImageGPTImageProcessorFast(BaseImageProcessorFast):
    """
    Constructs a fast ImageGPT image processor.
    
    This processor can be used to resize images to a smaller resolution (such as 32x32 or 64x64),
    normalize them and finally color quantize them to obtain sequences of "pixel values" (color clusters).
    """
    
    model_input_names = ["input_ids"]
    
    # Defaults largely aligned with the slow processor, except normalization which we do manually to [-1, 1]
    resample = PILImageResampling.BILINEAR
    size = {"height": 256, "width": 256}
    do_resize = True
    # we can't utilizye basefastimage processor normalization/rescale as ImageGPT uses (x/127.5 - 1) or [-1,1] normalization
    do_rescale = False
    do_normalize = False

    do_color_quantize = True
    clusters = None  # Must be set at instantiation
    
    def __init__(
        self,
        clusters: Optional[Union[list, torch.Tensor]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Convert clusters to torch tensor if provided
        if clusters is not None:
            if not isinstance(clusters, torch.Tensor):
                clusters = torch.tensor(clusters, dtype=torch.float32)
            self.clusters = clusters
        else:
            self.clusters = None

    def _preprocess(
        self,
        images,  # List[torch.Tensor], each (C, H, W)
        do_color_quantize: Optional[bool] = None,
        clusters: Optional[Union[list, torch.Tensor]] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ):
        # Ensure the base class does resize/crop only
       
        base_batch = super()._preprocess(images, return_tensors=return_tensors, **kwargs)
        pixel_values = base_batch["pixel_values"]  # Tensor [B,C,H,W] or list of [C,H,W]

        # Convert to float and apply ImageGPT normalization: [-1, 1]
        if isinstance(pixel_values, torch.Tensor):
            normalized = pixel_values.to(dtype=torch.float32) / 127.5 - 1.0
        else:
            normalized = [img.to(dtype=torch.float32) / 127.5 - 1.0 for img in pixel_values]

        # If color quantization is requested, perform it; otherwise return normalized pixel values
        do_color_quantize = do_color_quantize if do_color_quantize is not None else self.do_color_quantize
        if do_color_quantize:
            # Prepare clusters
            clusters = clusters if clusters is not None else self.clusters
            if clusters is None:
                raise ValueError("Clusters must be provided for color quantization.")
            if not isinstance(clusters, torch.Tensor):
                clusters = torch.tensor(clusters, dtype=torch.float32)

            # Quantize each image to a flattened sequence (H*W,)
            input_ids_list = []
            if isinstance(normalized, torch.Tensor):
                batch = normalized
                for img in batch:
                    device_clusters = clusters.to(img.device, dtype=img.dtype)
                    hwc = img.permute(1, 2, 0) 
                    flat = hwc.reshape(-1, 3)
                    ids = color_quantize_torch(flat, device_clusters)
                    input_ids_list.append(ids)
                input_ids = torch.stack(input_ids_list, dim=0)
            else:
                for img in normalized:
                    device_clusters = clusters.to(img.device, dtype=img.dtype)
                    hwc = img.permute(1, 2, 0)
                    flat = hwc.reshape(-1, 3)
                    ids = color_quantize_torch(flat, device_clusters)
                    input_ids_list.append(ids)
                input_ids = input_ids_list if return_tensors is None else torch.stack(input_ids_list, dim=0)

            from ...image_processing_utils import BatchFeature
            return BatchFeature(data={"input_ids": input_ids}, tensor_type=return_tensors)

        # Otherwise, return normalized pixel values
        base_batch["pixel_values"] = normalized
        return base_batch


__all__ = ["ImageGPTImageProcessorFast"]
