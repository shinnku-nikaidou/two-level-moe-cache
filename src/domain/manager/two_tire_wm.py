"""
Two-Tire Watermark expert cache manager.

This module provides a Python wrapper around the Rust-based TwoTireWmExpertCacheManager
for watermark-based expert caching with dual-tier memory management.
"""

from typing import List, Dict, Optional, Any
from ..cache.interfaces.expert_cache import IExpertCacheManager
from ..cache.entities.expert import Expert
from ..cache.entities.types import ExpertKey
from .. import ModelType

# Import the Rust implementation
try:
    from two_level_moe_cache.core import TwoTireWmExpertCacheManager as RustTwoTireWm
    from two_level_moe_cache.core import WatermarkConfig, ExpertRef
    from two_level_moe_cache.core import ExpertKey as RustExpertKey
    from two_level_moe_cache.core import ExpertParamType as RustExpertParamType
    from two_level_moe_cache.core import ModelType as RustModelType
    RUST_AVAILABLE = True
    print("✅ Rust core library loaded successfully")
except ImportError as e:
    print(f"⚠️  Rust core library not available: {e}")
    RUST_AVAILABLE = False
    RustTwoTireWm = None
    WatermarkConfig = None
    ExpertRef = None
    RustExpertKey = None
    RustExpertParamType = None
    RustModelType = None


class TwoTireWmExpertCacheManager(IExpertCacheManager):
    """
    Two-Tier Watermark-based expert cache manager.
    
    This implementation uses the watermark algorithm for dual-tier caching
    with VRAM and RAM memory tiers, managing experts based on benefit density
    and adaptive watermark thresholds.
    
    Key Features:
    - Pure watermark algorithm implementation in Rust
    - Receives fused predictions from external policy components
    - Adaptive dual-tier memory management
    - Benefit density-based caching decisions
    """
    
    def __init__(
        self, 
        model_type: ModelType,
        vram_capacity_mb: int = 512,
        ram_capacity_mb: int = 2048,
        vram_learning_rate: float = 0.01,
        ram_learning_rate: float = 0.01,
        **kwargs
    ):
        """
        Initialize two-tier watermark cache manager.
        
        Args:
            model_type: Type of model for configuration
            vram_capacity_mb: VRAM capacity limit in MB
            ram_capacity_mb: RAM capacity limit in MB
            vram_learning_rate: Learning rate for VRAM watermark updates
            ram_learning_rate: Learning rate for RAM watermark updates
            **kwargs: Additional configuration parameters
            
        Raises:
            RuntimeError: If Rust core library is not available
        """
        if not RUST_AVAILABLE:
            raise RuntimeError(
                "Rust core library not available. Please ensure the core library is compiled."
            )
            
        super().__init__(model_type)
        
        # Create watermark configuration
        config = WatermarkConfig(
            vram_capacity=vram_capacity_mb * 1024 * 1024,  # Convert to bytes
            ram_capacity=ram_capacity_mb * 1024 * 1024,
            vram_learning_rate=vram_learning_rate,
            ram_learning_rate=ram_learning_rate,
        )
        
        # Initialize Rust implementation
        self._rust_cache = RustTwoTireWm(
            model_type=self._rust_model_type(model_type),
            config=config,
            total_layers=self._config.num_hidden_layers
        )
        
        # Cache for tracking accessed experts
        self._accessed_experts = set()
    
    def update_fused_predictions(self, predictions: Dict[ExpertKey, float]) -> None:
        """
        Update fused predictions from external policy components.
        
        Args:
            predictions: Dictionary mapping ExpertKey to activation probability
        """
        # Convert ExpertKey to string format expected by Rust
        rust_predictions = {}
        for key, prob in predictions.items():
            key_str = self._expert_key_to_string(key)
            rust_predictions[key_str] = prob
        
        self._rust_cache.update_fused_predictions(rust_predictions)
    
    def get(self, key: ExpertKey) -> Expert:
        """
        Get expert by key, loading if necessary.
        
        Args:
            key: Expert identifier
            
        Returns:
            Expert instance with loaded weights
            
        Raises:
            KeyError: If expert cannot be loaded
        """
        # Convert our ExpertKey to Rust format
        rust_key = self._python_to_rust_key(key)
        rust_expert_ref = self._rust_cache.get(rust_key)
        
        # Get the Python Expert instance and update it
        expert = self._get_expert(key)
        self._sync_expert_from_rust(expert, rust_expert_ref)
        
        # Track access
        self._accessed_experts.add(key)
        
        return expert
    
    def get_batch(self, keys: List[ExpertKey]) -> List[Expert]:
        """
        Get batch of experts efficiently.
        
        Args:
            keys: List of expert identifiers
            
        Returns:
            List of expert instances in the same order as keys
            
        Raises:
            KeyError: If any expert cannot be loaded
        """
        rust_keys = [self._python_to_rust_key(key) for key in keys]
        rust_expert_refs = self._rust_cache.get_batch(rust_keys)
        
        experts = []
        for key, rust_ref in zip(keys, rust_expert_refs):
            expert = self._get_expert(key)
            self._sync_expert_from_rust(expert, rust_ref)
            experts.append(expert)
            self._accessed_experts.add(key)
        
        return experts
    
    def clear(self) -> None:
        """Clear all cached experts and reset state."""
        self._rust_cache.clear()
        self._accessed_experts.clear()
    
    def next(self) -> None:
        """Advance to next time step and apply watermark decisions."""
        self._rust_cache.next()
    
    def get_watermark_stats(self) -> Dict[str, Any]:
        """
        Get watermark statistics for monitoring and debugging.
        
        Returns:
            Dictionary with watermark values, capacity usage, and other stats
        """
        vram_wm, ram_wm = self._rust_cache.get_watermarks()
        (vram_used, vram_capacity), (ram_used, ram_capacity) = self._rust_cache.get_capacity_usage()
        stats = self._rust_cache.get_stats()
        
        return {
            "vram_watermark": vram_wm,
            "ram_watermark": ram_wm,
            "vram_usage_bytes": vram_used,
            "vram_capacity_bytes": vram_capacity,
            "vram_utilization": vram_used / vram_capacity if vram_capacity > 0 else 0.0,
            "ram_usage_bytes": ram_used,
            "ram_capacity_bytes": ram_capacity,
            "ram_utilization": ram_used / ram_capacity if ram_capacity > 0 else 0.0,
            "accessed_experts_count": len(self._accessed_experts),
            **stats
        }
    
    # Private helper methods
    def _python_to_rust_key(self, key: ExpertKey) -> RustExpertKey:
        """Convert Python ExpertKey to Rust ExpertKey."""
        return RustExpertKey(
            layer_idx=key.layer_idx,
            expert_id=key.expert_id,
            param_type=self._python_to_rust_param_type(key.param_type)
        )
    
    def _python_to_rust_param_type(self, param_type) -> RustExpertParamType:
        """Convert Python ExpertParamType to Rust ExpertParamType."""
        # Import the Python ExpertParamType enum
        from ..cache.entities.types import ExpertParamType
        
        mapping = {
            ExpertParamType.MLP1_WEIGHT: RustExpertParamType.MLP1_WEIGHT,
            ExpertParamType.MLP1_BIAS: RustExpertParamType.MLP1_BIAS,
            ExpertParamType.MLP2_WEIGHT: RustExpertParamType.MLP2_WEIGHT,
            ExpertParamType.MLP2_BIAS: RustExpertParamType.MLP2_BIAS,
        }
        return mapping[param_type]
    
    def _rust_model_type(self, model_type: ModelType):
        """Convert Python ModelType to Rust ModelType."""
        if not RUST_AVAILABLE:
            return None
            
        mapping = {
            ModelType.GPT_OSS_20B: RustModelType.GPT_OSS_20B,
            ModelType.GPT_OSS_120B: RustModelType.GPT_OSS_120B,
            ModelType.PHI_TINY_MOE: RustModelType.PHI_TINY_MOE,
        }
        return mapping[model_type]
    
    def _expert_key_to_string(self, key: ExpertKey) -> str:
        """Convert ExpertKey to string format expected by Rust."""
        return f"L{key.layer_idx}_E{key.expert_id}_{key.param_type}"
    
    def _sync_expert_from_rust(self, expert: Expert, rust_ref: ExpertRef) -> None:
        """Synchronize Python Expert state from Rust reference."""
        # Update expert's memory tier based on rust reference
        if rust_ref.tier is not None:
            from ..cache.entities.types import MemoryTier
            
            # Convert Rust MemoryTier to Python MemoryTier
            tier_mapping = {
                0: MemoryTier.VRAM,  # RustMemoryTier.VRAM
                1: MemoryTier.RAM,   # RustMemoryTier.RAM  
                2: MemoryTier.DISK,  # RustMemoryTier.DISK
            }
            expert.current_tier = tier_mapping.get(rust_ref.tier, MemoryTier.DISK)
        
        # Note: In a full implementation, we would also:
        # - Load actual tensor data if needed
        # - Update expert's data pointer
        # - Sync memory usage tracking
        # For now, we just update the tier information