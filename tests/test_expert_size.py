#!/usr/bin/env python3
"""
Test to calculate actual expert sizes for different MoE models.

This test loads model checkpoints using safetensors to analyze the actual
memory footprint of expert parameters in GPT-OSS models, and validates
the accuracy of Rust core assumptions.
"""

import os
from pathlib import Path
import re
from safetensors.torch import load_file


class TestExpertSizes:
    """Test expert memory consumption by analyzing actual model checkpoints."""

    def get_model_path(self, model_name):
        """Get path to model checkpoint directory."""
        base_path = Path("data/models")
        if model_name == "gpt-oss-20b":
            return base_path / "gpt-oss-20b" / "original"
        elif model_name == "gpt-oss-120b":
            return base_path / "gpt-oss-120b" / "original"
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def load_model_weights(self, model_path):
        """Load all safetensors files from model directory."""
        model_path = Path(model_path)
        weights = {}
        
        # Find all .safetensors files
        safetensor_files = list(model_path.glob("*.safetensors"))
        print(f"Found {len(safetensor_files)} safetensors files")
        
        # Load each file
        for file_path in safetensor_files:
            try:
                file_weights = load_file(str(file_path))
                weights.update(file_weights)
                print(f"Loaded {len(file_weights)} tensors from {file_path.name}")
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
        
        print(f"Total tensors loaded: {len(weights)}")
        return weights

    def examine_model_structure(self, weights, model_name="gpt-oss-20b"):
        """Examine and categorize model parameter structure."""
        print(f"\n=== {model_name.upper()} Model Structure Analysis ===")
        
        # Group parameters by pattern
        patterns = {}
        mlp_params = []
        
        for name, tensor in weights.items():
            # Look for MLP-related parameters
            if "mlp" in name.lower():
                mlp_params.append((name, tensor.shape, tensor.numel()))
            
            # Categorize by structure
            parts = name.split('.')
            if len(parts) >= 3:
                pattern = '.'.join(parts[:3])
                if pattern not in patterns:
                    patterns[pattern] = []
                patterns[pattern].append(name)
        
        print(f"Parameter patterns:")
        for pattern, names in sorted(patterns.items()):
            if "mlp" in pattern.lower():  # Only show MLP patterns for brevity
                print(f"  {pattern}: {len(names)} parameters")
                # Show first example with details
                if names and names[0] in weights:
                    example = names[0]
                    shape = list(weights[example].shape)
                    numel = weights[example].numel()
                    size_mb = numel * 4 / (1024 * 1024)
                    print(f"    Example: {example} {shape} ({size_mb:.2f} MB)")
        
        return mlp_params

    def calculate_expert_size_precise(self, weights, model_name="gpt-oss-20b"):
        """Calculate precise expert size from actual model parameters."""
        print(f"\n=== {model_name.upper()} Expert Size Analysis ===")
        
        # Find first layer's MLP parameters
        layer_0_mlp_params = {
            name: tensor for name, tensor in weights.items()
            if name.startswith("block.0.mlp.")
        }
        
        if not layer_0_mlp_params:
            print("No MLP parameters found for layer 0!")
            return None
        
        print(f"Analyzing Layer 0 MLP structure:")
        
        total_layer_bytes = 0
        total_layer_params = 0
        expert_count = None
        
        # Analyze each parameter
        for name, tensor in layer_0_mlp_params.items():
            shape = list(tensor.shape)
            numel = tensor.numel()
            size_bytes = numel * 4  # Assume float32
            size_mb = size_bytes / (1024 * 1024)
            
            print(f"  {name:30}: {str(shape):20} ({numel:>12,} params, {size_mb:>8.2f} MB)")
            
            # Detect expert count from first dimension of bias parameters
            if "bias" in name and expert_count is None:
                expert_count = shape[0]
            
            total_layer_bytes += size_bytes
            total_layer_params += numel
        
        # Summary
        total_layer_mb = total_layer_bytes / (1024 * 1024)
        print(f"  {'='*30} {'='*20} {'='*15} {'='*12}")
        print(f"  {'TOTAL LAYER':30}: {'':<20} ({total_layer_params:>12,} params, {total_layer_mb:>8.2f} MB)")
        
        if expert_count:
            # Calculate per-expert metrics
            expert_size_bytes = total_layer_bytes // expert_count
            expert_size_mb = expert_size_bytes / (1024 * 1024)
            expert_params = total_layer_params // expert_count
            
            print(f"\n📊 Expert Analysis:")
            print(f"  Detected expert count: {expert_count}")
            print(f"  Size per expert: {expert_size_mb:.2f} MB ({expert_params:,} params)")
            print(f"  Size per expert: {expert_size_bytes:,} bytes")
            
            return {
                'expert_count': expert_count,
                'expert_size_bytes': expert_size_bytes,
                'expert_size_mb': expert_size_mb,
                'expert_params': expert_params,
                'layer_total_mb': total_layer_mb,
                'layer_total_params': total_layer_params
            }
        
        return None

    def validate_rust_assumptions(self, expert_info):
        """Validate Rust core expert size assumptions."""
        if not expert_info:
            return
        
        expert_size_mb = expert_info['expert_size_mb']
        expert_size_bytes = expert_info['expert_size_bytes']
        
        # Current Rust assumptions (updated values)
        rust_assumptions = {
            'gpt-oss-20b': 52_923_244,    # ~50.47 MB
            'gpt-oss-120b': 100_000_000,  # ~95.37 MB estimated
            'phi-tiny-moe': 1_048_576,    # ~1 MB
        }
        
        print(f"\n🔍 Rust Core Validation:")
        for model, assumed_bytes in rust_assumptions.items():
            assumed_mb = assumed_bytes / (1024 * 1024)
            if 'gpt-oss-20b' in model.lower():  # Only validate for current model
                print(f"  {model}: {assumed_mb:.2f} MB assumption")
                ratio = expert_size_mb / assumed_mb
                print(f"  Actual vs assumption: {expert_size_mb:.2f} MB vs {assumed_mb:.2f} MB")
                print(f"  Accuracy ratio: {ratio:.2f}x")
                
                if abs(ratio - 1.0) < 0.1:
                    print(f"  ✅ Excellent accuracy (within 10%)")
                elif abs(ratio - 1.0) < 0.2:
                    print(f"  ✅ Good accuracy (within 20%)")
                elif abs(ratio - 1.0) < 0.5:
                    print(f"  ⚠️  Moderate accuracy (within 50%)")
                else:
                    print(f"  🚨 Poor accuracy (>50% difference)")
                    if ratio > 1.5:
                        print(f"      Rust assumption too low, increase to ~{expert_size_bytes}")
                    else:
                        print(f"      Rust assumption too high, decrease to ~{expert_size_bytes}")

    def calculate_memory_scenarios(self, expert_info):
        """Calculate memory usage scenarios for different cache sizes."""
        if not expert_info:
            return
        
        expert_size_bytes = expert_info['expert_size_bytes']
        
        # Estimate total model size
        total_layers = 24  # Based on GPT-OSS structure
        experts_per_layer = expert_info['expert_count']
        total_experts = total_layers * experts_per_layer
        total_model_gb = (expert_size_bytes * total_experts) / (1024**3)
        
        print(f"\n🏗️  Full Model Statistics:")
        print(f"  Total layers: {total_layers}")
        print(f"  Experts per layer: {experts_per_layer}")
        print(f"  Total experts: {total_experts}")
        print(f"  Total expert weights: {total_model_gb:.1f} GB")
        
        # Memory usage scenarios
        print(f"\n💾 Cache Memory Scenarios:")
        cache_scenarios = [
            ("Minimal", 8),
            ("Small", 16),
            ("Medium", 32),
            ("Large", 64),
            ("XLarge", 128),
            ("XXLarge", 256)
        ]
        
        for scenario_name, expert_count in cache_scenarios:
            if expert_count <= total_experts:
                cache_gb = (expert_count * expert_size_bytes) / (1024**3)
                percent = (expert_count / total_experts) * 100
                print(f"  {scenario_name:8} ({expert_count:3} experts): {cache_gb:5.1f} GB ({percent:4.1f}% of model)")

    def test_gpt_oss_20b_expert_size(self):
        """Test expert size calculation for GPT-OSS 20B model."""
        model_path = self.get_model_path("gpt-oss-20b")
        
        if not model_path.exists():
            print(f"Model path not found: {model_path}")
            print("Please ensure GPT-OSS 20B model is downloaded to data/models/gpt-oss-20b/original/")
            return None
        
        weights = self.load_model_weights(model_path)
        
        # Structure analysis
        self.examine_model_structure(weights, "gpt-oss-20b")
        
        # Precise size calculation
        expert_info = self.calculate_expert_size_precise(weights, "gpt-oss-20b")
        
        if expert_info:
            # Validate Rust assumptions
            self.validate_rust_assumptions(expert_info)
            
            # Memory scenarios
            self.calculate_memory_scenarios(expert_info)
        
        return expert_info

    def test_gpt_oss_120b_expert_size(self):
        """Test expert size calculation for GPT-OSS 120B model."""
        model_path = self.get_model_path("gpt-oss-120b")
        
        if not model_path.exists():
            print(f"Model path not found: {model_path}")
            print("Please ensure GPT-OSS 120B model is downloaded to data/models/gpt-oss-120b/original/")
            return None
        
        weights = self.load_model_weights(model_path)
        expert_info = self.calculate_expert_size_precise(weights, "gpt-oss-120b")
        
        if expert_info:
            self.validate_rust_assumptions(expert_info)
            self.calculate_memory_scenarios(expert_info)
        
        return expert_info

    def test_compare_models(self):
        """Compare expert sizes between different models."""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        models = ["gpt-oss-20b"]  # Add "gpt-oss-120b" when available
        results = {}
        
        for model_name in models:
            try:
                if model_name == "gpt-oss-20b":
                    result = self.test_gpt_oss_20b_expert_size()
                elif model_name == "gpt-oss-120b":
                    result = self.test_gpt_oss_120b_expert_size()
                else:
                    continue
                    
                if result:
                    results[model_name] = result
            except Exception as e:
                print(f"Failed to analyze {model_name}: {e}")
        
        # Summary comparison
        if len(results) > 1:
            print(f"\n📊 Model Size Comparison:")
            for model_name, result in results.items():
                print(f"  {model_name:15}: {result['expert_size_mb']:6.1f} MB per expert")
        
        return results

    def test_rust_integration_readiness(self):
        """Test if the codebase is ready for Rust-Python integration."""
        print(f"\n🔧 Rust Integration Readiness Check:")
        
        # Check if expert size calculations are reasonable for cache scenarios
        result = self.test_gpt_oss_20b_expert_size()
        
        if result:
            expert_mb = result['expert_size_mb']
            
            # Typical cache scenarios
            scenarios = [
                ("8GB VRAM", 8 * 1024, "entry-level GPU"),
                ("16GB VRAM", 16 * 1024, "mid-range GPU"), 
                ("24GB VRAM", 24 * 1024, "high-end GPU"),
                ("32GB RAM", 32 * 1024, "typical system RAM"),
                ("64GB RAM", 64 * 1024, "large system RAM")
            ]
            
            print(f"\n💾 Practical Cache Capacity Analysis:")
            print(f"   (Expert size: {expert_mb:.1f} MB each)")
            
            for name, capacity_mb, description in scenarios:
                max_experts = int(capacity_mb / expert_mb)
                print(f"  {name:10} ({description:15}): max {max_experts:3} experts")
                
                if max_experts < 8:
                    print(f"            ⚠️  Very limited ({max_experts} experts)")
                elif max_experts < 32:
                    print(f"            ✅ Workable ({max_experts} experts)")
                else:
                    print(f"            🚀 Excellent ({max_experts} experts)")
        
        return result


if __name__ == "__main__":
    # Run comprehensive expert size analysis
    test_instance = TestExpertSizes()
    
    print("🔬 GPT-OSS Expert Size Analysis")
    print("=" * 50)
    
    # Primary test
    results = test_instance.test_compare_models()
    
    # Integration readiness check
    test_instance.test_rust_integration_readiness()
    
    print(f"\n✅ Analysis complete!")
    if results:
        for model, result in results.items():
            print(f"   {model}: {result['expert_size_mb']:.1f} MB per expert")