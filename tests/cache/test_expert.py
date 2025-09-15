"""
Unit tests for Expert class and caching entities.
"""

import sys
import os
import torch
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.domain.cache.entities.expert import Expert, ExpertKey, MemoryTier


class TestExpertKey:
    """Test ExpertKey functionality."""
    
    def test_expert_key_creation(self):
        """Test creating ExpertKey with valid parameters."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        assert key.layer_idx == 0
        assert key.expert_id == 5
        assert key.param_type == "mlp1_weight"
    
    def test_expert_key_string_representation(self):
        """Test string representation of ExpertKey."""
        key = ExpertKey(layer_idx=12, expert_id=31, param_type="mlp2_bias")
        assert str(key) == "L12_E31_mlp2_bias"
    
    def test_expert_key_hashing(self):
        """Test that ExpertKey can be used as dict key."""
        key1 = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        key2 = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        key3 = ExpertKey(layer_idx=0, expert_id=6, param_type="mlp1_weight")
        
        # Same keys should have same hash
        assert hash(key1) == hash(key2)
        # Different keys should have different hash (usually)
        assert hash(key1) != hash(key3)
        
        # Should work as dict keys
        cache = {key1: "value1", key3: "value3"}
        assert cache[key2] == "value1"  # key2 == key1


class TestMemoryTier:
    """Test MemoryTier enum."""
    
    def test_memory_tier_values(self):
        """Test MemoryTier enum values for ordering."""
        assert MemoryTier.VRAM.value == 0  # Fastest
        assert MemoryTier.RAM.value == 1   # Medium
        assert MemoryTier.DISK.value == 2  # Slowest
    
    def test_memory_tier_ordering(self):
        """Test that tier comparison works correctly."""
        assert MemoryTier.VRAM.value < MemoryTier.RAM.value
        assert MemoryTier.RAM.value < MemoryTier.DISK.value


class TestExpert:
    """Test Expert class functionality."""
    
    def test_expert_initialization(self):
        """Test Expert initialization with default values."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        expert = Expert(key)
        
        assert expert.expert_key == key
        assert expert.current_tier == MemoryTier.DISK
        assert expert.data is None
        assert expert.device is None
        assert expert.access_count == 0
        assert not expert.is_loaded
        assert expert.memory_usage() == 0
    
    def test_expert_data_loading(self):
        """Test loading data into Expert."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        expert = Expert(key)
        
        # Load data
        tensor = torch.randn(1024, 512, dtype=torch.float32)
        expert.data = tensor
        
        assert expert.is_loaded
        assert expert.access_count == 1  # data setter increments access count
        assert expert.memory_usage() == tensor.numel() * tensor.element_size()
        assert torch.equal(expert.data, tensor)
    
    def test_expert_tier_promotion(self):
        """Test promoting expert to higher tier."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        expert = Expert(key, MemoryTier.DISK)
        
        # Promote to RAM
        expert.promote_to(MemoryTier.RAM)
        assert expert.current_tier == MemoryTier.RAM
        
        # Promote to VRAM
        expert.promote_to(MemoryTier.VRAM)
        assert expert.current_tier == MemoryTier.VRAM
        
        # Cannot "promote" to lower tier
        expert.promote_to(MemoryTier.DISK)
        assert expert.current_tier == MemoryTier.VRAM  # Should remain unchanged
    
    def test_expert_tier_demotion(self):
        """Test demoting expert to lower tier."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        expert = Expert(key, MemoryTier.VRAM)
        
        # Demote to RAM
        expert.demote_to(MemoryTier.RAM)
        assert expert.current_tier == MemoryTier.RAM
        
        # Demote to DISK
        expert.demote_to(MemoryTier.DISK)
        assert expert.current_tier == MemoryTier.DISK
        
        # Cannot "demote" to higher tier
        expert.demote_to(MemoryTier.VRAM)
        assert expert.current_tier == MemoryTier.DISK  # Should remain unchanged
    
    def test_expert_device_movement(self):
        """Test moving expert data to different devices."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        expert = Expert(key)
        
        # Load data on CPU
        tensor = torch.randn(100, 50, dtype=torch.float32)
        expert.data = tensor
        assert expert.data.device == torch.device("cpu")
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            expert.move_to_device("cuda")
            assert expert.data.device.type == "cuda"
            assert expert.device is not None and expert.device.type == "cuda"
    
    def test_expert_unloading(self):
        """Test unloading expert data."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        expert = Expert(key)
        
        # Load data
        tensor = torch.randn(100, 50, dtype=torch.float32)
        expert.data = tensor
        assert expert.is_loaded
        
        # Unload data
        expert.unload()
        assert not expert.is_loaded
        assert expert.data is None
        assert expert.memory_usage() == 0
    
    def test_expert_access_tracking(self):
        """Test access time and count tracking."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        expert = Expert(key)
        
        initial_time = expert.last_access_time
        initial_count = expert.access_count
        
        # Touch expert
        expert.touch()
        
        assert expert.access_count == initial_count + 1
        assert expert.last_access_time > initial_time
    
    def test_expert_string_representation(self):
        """Test Expert string representation."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        expert = Expert(key, MemoryTier.RAM)
        
        # Unloaded state
        repr_str = str(expert)
        assert "L0_E5_mlp1_weight" in repr_str
        assert "@1" in repr_str  # RAM tier
        assert "unloaded" in repr_str
        assert "accessed: 0" in repr_str
        
        # Loaded state
        tensor = torch.randn(10, 20)
        expert.data = tensor
        repr_str = str(expert)
        assert "loaded" in repr_str
        assert str(tensor.shape) in repr_str


def test_expert_workflow():
    """Test complete Expert workflow."""
    # Create expert key
    key = ExpertKey(layer_idx=5, expert_id=15, param_type="mlp2_weight")
    
    # Create expert starting on disk
    expert = Expert(key, MemoryTier.DISK)
    assert expert.current_tier == MemoryTier.DISK
    assert not expert.is_loaded
    
    # Load data and promote to RAM
    tensor = torch.randn(2048, 1024, dtype=torch.float32)
    expert.promote_to(MemoryTier.RAM)
    expert.data = tensor
    assert expert.current_tier == MemoryTier.RAM
    assert expert.is_loaded
    assert expert.memory_usage() == 2048 * 1024 * 4  # float32 = 4 bytes
    
    # Promote to VRAM if CUDA available
    if torch.cuda.is_available():
        expert.move_to_device("cuda")
        expert.promote_to(MemoryTier.VRAM)
        assert expert.current_tier == MemoryTier.VRAM
        assert expert.data.device.type == "cuda"
    
    # Access tracking
    initial_count = expert.access_count
    expert.touch()
    assert expert.access_count == initial_count + 1
    
    # Demote back to disk
    expert.demote_to(MemoryTier.DISK)
    expert.unload()
    assert expert.current_tier == MemoryTier.DISK
    assert not expert.is_loaded
    assert expert.memory_usage() == 0


if __name__ == "__main__":
    # Run basic tests if called directly
    test_expert_workflow()
    print("All basic tests passed!")