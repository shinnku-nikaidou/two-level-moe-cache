"""
Example usage of Expert with ExpertAdapter.
"""

import torch
from src.domain import ModelType
from src.domain.cache.entities.expert import Expert, ExpertKey, MemoryTier
from src.adapters.expert.factory import AdapterFactory


def example_expert_with_adapter():
    """Demonstrate Expert class with ExpertAdapter usage."""
    
    # Create adapter for GPT-OSS-20B
    model_type = ModelType.GPT_OSS_20B
    checkpoint_path = "data/models/gpt-oss-20b/original/"
    
    try:
        adapter = AdapterFactory.create_adapter(model_type, checkpoint_path)
        print(f"Created adapter: {adapter.__class__.__name__}")
    except (ValueError, FileNotFoundError) as e:
        print(f"Cannot create adapter: {e}")
        return
    
    # Create expert key for a specific weight
    expert_key = ExpertKey(layer_idx=0, expert_id=5, param_type="mlp1_weight")
    print(f"Created expert key: {expert_key}")
    
    # Create expert with adapter
    expert = Expert(
        expert_key=expert_key,
        model_type=model_type,
        adapter=adapter,
        current_tier=MemoryTier.DISK
    )
    print(f"Initial expert: {expert}")
    
    # Load from NVMe to RAM
    print("\n--- Loading from NVMe to RAM ---")
    try:
        expert.load_from_nvme_to_ram()
        print(f"After loading to RAM: {expert}")
        print(f"Tensor shape: {expert.data.shape if expert.data is not None else 'No data'}")
        print(f"Memory usage: {expert.memory_usage()} bytes")
    except Exception as e:
        print(f"Failed to load to RAM: {e}")
        return
    
    # Promote to VRAM if CUDA available
    if torch.cuda.is_available():
        print("\n--- Promoting to VRAM ---")
        try:
            expert.self_promote()  # RAM -> VRAM
            print(f"After promoting to VRAM: {expert}")
            if expert.data is not None:
                print(f"Tensor device: {expert.data.device}")
        except Exception as e:
            print(f"Failed to promote to VRAM: {e}")
    
    # Demote back through the tiers
    print("\n--- Demoting through tiers ---")
    expert.self_demote()  # VRAM -> RAM or RAM -> DISK
    print(f"After first demotion: {expert}")
    
    expert.self_demote()  # RAM -> DISK
    print(f"After second demotion: {expert}")
    
    print(f"Final memory usage: {expert.memory_usage()} bytes")


def example_direct_vram_loading():
    """Demonstrate direct loading from NVMe to VRAM."""
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping direct VRAM loading example")
        return
    
    # Create adapter and expert
    model_type = ModelType.GPT_OSS_20B
    checkpoint_path = "data/models/gpt-oss-20b/original/"
    
    try:
        adapter = AdapterFactory.create_adapter(model_type, checkpoint_path)
        expert_key = ExpertKey(layer_idx=1, expert_id=10, param_type="mlp2_bias")
        
        expert = Expert(
            expert_key=expert_key,
            model_type=model_type,
            adapter=adapter
        )
        
        print(f"\n--- Direct NVMe to VRAM Loading ---")
        print(f"Initial expert: {expert}")
        
        expert.load_from_nvme_to_vram()
        print(f"After direct VRAM loading: {expert}")
        if expert.data is not None:
            print(f"Tensor device: {expert.data.device}")
            print(f"Tensor shape: {expert.data.shape}")
            
    except Exception as e:
        print(f"Failed direct VRAM loading: {e}")


if __name__ == "__main__":
    print("=== Expert with ExpertAdapter Example ===")
    example_expert_with_adapter()
    example_direct_vram_loading()
    print("\n=== Example Complete ===")