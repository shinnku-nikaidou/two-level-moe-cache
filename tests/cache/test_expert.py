"""
Unit tests for Expert class and caching entities.
"""

import sys
import os
import torch
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.domain.cache.entities import Expert, ExpertKey, MemoryTier
from src.domain import ModelType


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

    def test_expert_key_validation(self):
        """Test ExpertKey parameter validation."""
        # Negative layer_idx should raise ValueError
        with pytest.raises(ValueError, match="layer_idx must be non-negative"):
            ExpertKey(layer_idx=-1, expert_id=5, param_type="mlp1_weight")

        # Negative expert_id should raise ValueError
        with pytest.raises(ValueError, match="expert_id must be non-negative"):
            ExpertKey(layer_idx=0, expert_id=-1, param_type="mlp1_weight")

        # Empty param_type should raise ValueError
        with pytest.raises(ValueError, match="param_type cannot be empty"):
            ExpertKey(layer_idx=0, expert_id=5, param_type="")


class TestMemoryTier:
    """Test MemoryTier enum."""

    def test_memory_tier_values(self):
        """Test MemoryTier enum values for ordering."""
        assert MemoryTier.VRAM.value == 0
        assert MemoryTier.RAM.value == 1
        assert MemoryTier.DISK.value == 2

    def test_memory_tier_ordering(self):
        """Test that memory tiers can be ordered by performance."""
        assert MemoryTier.VRAM.value < MemoryTier.RAM.value
        assert MemoryTier.RAM.value < MemoryTier.DISK.value


class TestExpert:
    """Test Expert class functionality."""

    def test_expert_initialization(self):
        """Test Expert initialization with default values."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        expert = Expert(key, ModelType.GPT_OSS_20B)

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
        expert = Expert(key, ModelType.GPT_OSS_20B)

        # Create some dummy data
        data = torch.randn(100, 200)
        expert.data = data
        expert.current_tier = MemoryTier.RAM

        assert expert.is_loaded
        assert expert.memory_usage() > 0
        assert expert.data.shape == (100, 200)

    def test_expert_tier_promotion(self):
        """Test promoting expert to higher tier."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        expert = Expert(key, ModelType.GPT_OSS_20B, current_tier=MemoryTier.DISK)

        # Mock promotion from DISK -> RAM (would normally load from disk)
        expert.data = torch.randn(10, 20)
        expert.current_tier = MemoryTier.RAM

        assert expert.is_loaded
        assert expert.current_tier == MemoryTier.RAM

        # Promote to VRAM (if CUDA available)
        if torch.cuda.is_available():
            expert.promote_to_upper_tier()
            assert expert.current_tier == MemoryTier.VRAM
            assert expert.device is not None

    def test_expert_tier_demotion(self):
        """Test demoting expert to lower tier."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        expert = Expert(key, ModelType.GPT_OSS_20B, current_tier=MemoryTier.RAM)
        expert.data = torch.randn(10, 20)

        # Demote to DISK
        expert.demote_to_lower_tier()
        assert expert.current_tier == MemoryTier.DISK
        assert expert.data is None
        assert not expert.is_loaded

    def test_expert_device_movement(self):
        """Test moving expert data to different devices."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        expert = Expert(key, ModelType.GPT_OSS_20B)
        expert.data = torch.randn(10, 20)
        expert.current_tier = MemoryTier.RAM

        # Move to RAM
        expert.move_to_ram()
        assert expert.current_tier == MemoryTier.RAM
        assert expert.device == torch.device("cpu")

        # Try to move to VRAM if CUDA is available
        if torch.cuda.is_available():
            expert.move_to_vram()
            assert expert.current_tier == MemoryTier.VRAM
            assert "cuda" in str(expert.device)

    def test_expert_unloading(self):
        """Test unloading expert data."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        expert = Expert(key, ModelType.GPT_OSS_20B)
        expert.data = torch.randn(10, 20)
        expert.current_tier = MemoryTier.RAM

        assert expert.is_loaded

        expert.unload()
        assert not expert.is_loaded
        assert expert.data is None
        assert expert.current_tier == MemoryTier.DISK

    def test_expert_access_tracking(self):
        """Test access time and count tracking."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        expert = Expert(key, ModelType.GPT_OSS_20B)

        initial_time = expert.last_access_time
        initial_count = expert.access_count

        # Update access
        expert.update_access_time()

        assert expert.last_access_time >= initial_time
        assert expert.access_count == initial_count + 1

    def test_expert_string_representation(self):
        """Test Expert string representation."""
        key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
        expert = Expert(key, ModelType.GPT_OSS_20B, current_tier=MemoryTier.RAM)
        expert.data = torch.randn(100, 200)

        repr_str = repr(expert)
        assert "Expert" in repr_str
        assert "L0_E5_mlp1_weight" in repr_str
        assert "loaded" in repr_str
        assert "RAM" in repr_str


def test_expert_workflow():
    """Test complete Expert workflow."""
    # Create expert key
    key = ExpertKey(layer_idx=5, expert_id=15, param_type="mlp2_weight")

    # Create expert starting on disk
    expert = Expert(key, ModelType.GPT_OSS_20B, current_tier=MemoryTier.DISK)

    assert expert.current_tier == MemoryTier.DISK
    assert not expert.is_loaded

    # Mock loading data (in real scenario would load from adapter)
    expert.data = torch.randn(256, 1024)
    expert.current_tier = MemoryTier.RAM

    assert expert.is_loaded
    assert expert.memory_usage() > 0

    # Try to move to VRAM if available
    if torch.cuda.is_available():
        expert.move_to_vram()
        assert expert.current_tier == MemoryTier.VRAM

        # Then demote back
        expert.demote_to_lower_tier()  # VRAM -> RAM
        assert expert.current_tier == MemoryTier.RAM

        expert.demote_to_lower_tier()  # RAM -> DISK
        assert expert.current_tier == MemoryTier.DISK
        assert not expert.is_loaded
