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
    
    # Default values matching the slow processor
    resample = PILImageResampling.BILINEAR
    size = {"height": 256, "width": 256} # import get_size_dict?, can be overridden in preprocess
    do_resize = True
    do_normalize = True
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
        **kwargs
    ):
        # Standard steps (resize, normalize, etc.) handled by base class
        images = super()._preprocess(images, **kwargs)  # List[torch.Tensor], (C, H, W)
        
        # Handle color quantization parameters
        do_color_quantize = do_color_quantize if do_color_quantize is not None else self.do_color_quantize
        clusters = clusters if clusters is not None else self.clusters
        
        if do_color_quantize:
            if clusters is None:
                raise ValueError("Clusters must be provided for color quantization.")
            
            # Ensure clusters is a torch tensor on the correct device
            if not isinstance(clusters, torch.Tensor):
                clusters = torch.tensor(clusters, dtype=torch.float32)
            
            input_ids = []
            for img in images:
                # Move clusters to the same device as the image
                clusters_device = clusters.to(img.device)
                
                # img: (C, H, W) -> (H, W, C)
                img = img.permute(1, 2, 0)
                # Flatten to (H*W, 3)
                flat_img = img.reshape(-1, 3)
                # Quantize pixels to cluster indices
                indices = color_quantize_torch(flat_img, clusters_device)
                # Reshape back to (H, W)
                quantized = indices.reshape(img.shape[0], img.shape[1])
                input_ids.append(quantized)
            
            # Return as BatchFeature with input_ids
            from ...image_processing_utils import BatchFeature
            return BatchFeature(data={"input_ids": input_ids})
        
        # If no color quantization, return pixel values as usual
        return images


__all__ = ["ImageGPTImageProcessorFast"]
