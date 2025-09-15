"""
Core entities for expert weight caching system.
"""

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, TYPE_CHECKING
import torch

from src.domain import ModelType
from src.adapters.expert.factory import AdapterFactory

if TYPE_CHECKING:
    from src.adapters.expert.base import ExpertAdapter


def get_checkpoint_path(model_type: ModelType) -> str:
    """
    Get the checkpoint path for a given model type.
    
    Args:
        model_type: The model type to get the checkpoint path for
        
    Returns:
        str: The path to the model checkpoint directory
        
    Raises:
        ValueError: If the model type is not supported or path doesn't exist
    """
    # Base models directory
    base_dir = "data/models"
    
    # Map model types to their directory names
    model_path_map = {
        ModelType.GPT_OSS_20B: "gpt-oss-20b/original/",
        ModelType.GPT_OSS_120B: "gpt-oss-120b/original/", 
        ModelType.PHI_TINY_MOE: "phi-tiny-moe/original/",
    }
    
    if model_type not in model_path_map:
        supported_models = list(model_path_map.keys())
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported models: {[m.value for m in supported_models]}"
        )
    
    checkpoint_path = os.path.join(base_dir, model_path_map[model_type])
    
    # Verify the path exists
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    return checkpoint_path


class MemoryTier(Enum):
    """Memory storage tiers for expert weights."""

    VRAM = 0  # GPU memory - fastest access
    RAM = 1  # CPU memory - medium access speed
    DISK = 2  # Disk storage - slowest but largest capacity


@dataclass(frozen=True)
class ExpertKey:
    """Unique identifier for expert weights."""

    layer_idx: int  # 0-23 (24 layers)
    expert_id: int  # 0-31 (32 experts)
    param_type: str  # "mlp1_weight" | "mlp1_bias" | "mlp2_weight" | "mlp2_bias"

    def __hash__(self):
        return hash((self.layer_idx, self.expert_id, self.param_type))

    def __str__(self):
        return f"L{self.layer_idx}_E{self.expert_id}_{self.param_type}"


class Expert:
    """
    Encapsulates expert weight data with tier-aware storage management.

    This class manages a single expert's weight data across different memory tiers
    (VRAM/RAM/DISK) and provides operations for tier migration and device management.
    """

    def __init__(
        self,
        expert_key: ExpertKey,
        model_type: ModelType,
        current_tier: MemoryTier = MemoryTier.DISK,
        data: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize expert weight data.

        Args:
            expert_key: Unique identifier for this expert weight
            model_type: The model type this expert belongs to
            current_tier: Current memory tier where data is stored
            data: The actual tensor data (None means not loaded)
            device: Device where the tensor should be placed when loaded
        """
        self.expert_key = expert_key
        self.model_type = model_type
        self.current_tier = current_tier
        self._data: Optional[torch.Tensor] = data
        self.device = torch.device(device) if device else None
        self.last_access_time = time.time()
        self.access_count = 0
        
        # Create adapter immediately based on model_type
        
        checkpoint_path = get_checkpoint_path(self.model_type)
        self._adapter: 'ExpertAdapter' = AdapterFactory.create_adapter(self.model_type, checkpoint_path)

    @property
    def adapter(self) -> 'ExpertAdapter':
        """Get the expert adapter."""
        return self._adapter

    @property
    def data(self) -> Optional[torch.Tensor]:
        """Get the tensor data if loaded in memory."""
        return self._data

    @data.setter
    def data(self, tensor: Optional[torch.Tensor]):
        """Set the tensor data and update access time."""
        self._data = tensor
        self.last_access_time = time.time()
        if tensor is not None:
            self.access_count += 1

    @property
    def is_loaded(self) -> bool:
        """Check if the tensor data is currently loaded in memory."""
        return self._data is not None

    def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        if self._data is None:
            return 0
        return self._data.numel() * self._data.element_size()

    def load_from_nvme_to_ram(self) -> None:
        """
        Load expert data from NVMe/disk to RAM.
        
        Raises:
            ValueError: If the expert_key is not supported by the adapter
        """
        if self.is_loaded:
            return  # Already loaded, no need to load again
        
        if self.current_tier != MemoryTier.DISK:
            return  # Not on disk, cannot load from NVMe
        
        # Load tensor using adapter
        tensor = self.adapter.load_expert_tensor(self.expert_key)
        
        # Place on CPU (RAM)
        self.data = tensor.to("cpu")
        self.current_tier = MemoryTier.RAM
    
    def load_from_nvme_to_vram(self) -> None:
        """
        Load expert data from NVMe/disk directly to VRAM.
        
        Raises:
            RuntimeError: If CUDA not available or expert is already loaded
            ValueError: If the expert_key is not supported by the adapter
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available, cannot load to VRAM")
        
        if self.is_loaded:
            return  # Already loaded, no need to load again
        
        if self.current_tier != MemoryTier.DISK:
            return  # Not on disk, cannot load from NVMe
        
        # Load tensor using adapter
        tensor = self.adapter.load_expert_tensor(self.expert_key)
        
        # Place on CUDA device (VRAM)
        cuda_device = "cuda" if self.device is None else self.device
        self.data = tensor.to(cuda_device)
        self.device = torch.device(cuda_device)
        self.current_tier = MemoryTier.VRAM

    def self_promote(self) -> None:
        """Promote to the next higher memory tier."""
        match self.current_tier:
            case MemoryTier.DISK:
                self.load_from_nvme_to_ram()
                self.current_tier = MemoryTier.RAM
            case MemoryTier.RAM:
                if self._data is not None and torch.cuda.is_available():
                    cuda_device = "cuda" if self.device is None else self.device
                    self._data = self._data.to(cuda_device)
                    self.device = torch.device(cuda_device)
                    self.current_tier = MemoryTier.VRAM
            case MemoryTier.VRAM:
                pass  # Already at highest tier
            
    def self_demote(self) -> None:
        match self.current_tier:
            case MemoryTier.VRAM:
                if self._data is not None:
                    self._data = self._data.to("cpu")
                    self.current_tier = MemoryTier.RAM
            case MemoryTier.RAM:
                self._data = None  # Unload from RAM, keep on DISK
                self.current_tier = MemoryTier.DISK
            case MemoryTier.DISK:
                pass  # Already at lowest tier

    # def move_to_device(self, device: Union[str, torch.device]) -> None:
    #     """
    #     Move the tensor data to a specific device.

    #     Args:
    #         device: Target device (e.g., 'cuda', 'cpu')
    #     """
    #     if self._data is not None:
    #         self.device = torch.device(device)
    #         self._data = self._data.to(device)

    def destroy(self) -> None:
        """Unload the tensor data from memory."""
        self._data = None

    def __repr__(self):
        status = f"@{self.current_tier.value}"
        if self.is_loaded:
            status += f" loaded({self._data.shape})"  # pyright: ignore[reportOptionalMemberAccess]
        else:
            status += " unloaded"
        return f"Expert[{self.expert_key}] {status} (accessed: {self.access_count})"
