"""
GPT-OSS weight precomputation infrastructure.

This module provides tools to precompute MXFP4 weights into FP16/BF16 format
for dramatically faster loading during inference. Each expert tensor is stored
as an individual safetensors file for true on-demand loading.

Total files for GPT-OSS-20B: 24 layers × 32 experts × 4 params = 3072 files
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from src.boilerplate.gpt_oss.weights import Checkpoint
from src.common.types import ExpertKey, ExpertParamType
from src.domain import ModelType


class GPTOSSPrecomputer:
    """GPT-OSS weight precomputer - generates 3072 individual files."""

    def __init__(
        self, model_type: ModelType, target_dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize precomputer for GPT-OSS models.

        Args:
            model_type: Only ModelType.GPT_OSS_20B is supported
            target_dtype: Target dtype for precomputed weights (bfloat16 recommended)
        """
        self.model_type = model_type
        self.target_dtype = target_dtype

        # Model configuration (only GPT-OSS-20B supported)
        if model_type == ModelType.GPT_OSS_20B:
            self.num_layers = 24
            self.num_experts = 32
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Calculate total expert files
        self.total_expert_files = (
            self.num_layers * self.num_experts * len(ExpertParamType)
        )
        print(f"Total expert files to generate: {self.total_expert_files}")

        # Initialize checkpoint for MXFP4 decoding
        from src.config.util import get_checkpoint_path

        self.checkpoint = Checkpoint(
            get_checkpoint_path(model_type), device=torch.device("cpu")
        )

        # Setup precomputed storage paths
        self.checkpoint_path = Path(get_checkpoint_path(model_type))
        self.precomputed_dir = self.checkpoint_path / "precomputed"
        self.precomputed_dir.mkdir(exist_ok=True)

        # Create layer directories
        for layer_idx in range(self.num_layers):
            layer_dir = self.precomputed_dir / f"layer_{layer_idx:02d}"
            layer_dir.mkdir(exist_ok=True)

    def precompute_all_experts(self, force_rebuild: bool = False) -> None:
        """
        Precompute all 3072 expert weight files.

        Args:
            force_rebuild: If True, rebuild existing files
        """
        print(f"Starting precomputation for {self.model_type.value}")
        print(f"Target dtype: {self.target_dtype}")
        print(f"Storage path: {self.precomputed_dir}")
        print(f"Generating {self.total_expert_files} individual safetensors files")

        # Generate metadata
        metadata = {
            "model_type": self.model_type.value,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "total_expert_files": self.total_expert_files,
            "target_dtype": str(self.target_dtype),
            "format_version": "1.0",
            "storage_format": "individual_files",
            "file_naming": "layer_{layer:02d}/expert_{expert:02d}_{param_type}.safetensors",
            "created_by": "GPTOSSPrecomputer",
        }

        # Statistics tracking
        processed_count = 0
        skipped_count = 0
        error_count = 0

        # Process all experts with progress bar
        with tqdm(total=self.total_expert_files, desc="Precomputing experts") as pbar:
            for layer_idx in range(self.num_layers):
                for expert_id in range(self.num_experts):
                    for param_type in ExpertParamType:

                        # Build file path
                        expert_file = self._get_expert_file_path(
                            layer_idx, expert_id, param_type
                        )

                        # Skip if already exists
                        if expert_file.exists() and not force_rebuild:
                            skipped_count += 1
                            pbar.update(1)
                            continue

                        # Precompute single expert weight
                        try:
                            expert_tensor = self._precompute_single_expert(
                                layer_idx, expert_id, param_type
                            )

                            # Save individual file
                            save_file({"tensor": expert_tensor}, expert_file)
                            processed_count += 1

                        except Exception as e:
                            print(
                                f"\n❌ Failed: layer_{layer_idx} expert_{expert_id} {param_type.value}: {e}"
                            )
                            error_count += 1

                        pbar.update(1)

        # Save metadata
        with open(self.precomputed_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Precomputation completed!")
        print(f"   Processed: {processed_count} files")
        print(f"   Skipped: {skipped_count} files")
        print(f"   Errors: {error_count} files")
        print(f"   Total: {processed_count + skipped_count} files")

    def _get_expert_file_path(
        self, layer_idx: int, expert_id: int, param_type: ExpertParamType
    ) -> Path:
        """Get file path for a specific expert tensor."""
        layer_dir = self.precomputed_dir / f"layer_{layer_idx:02d}"
        filename = f"expert_{expert_id:02d}_{param_type.value}.safetensors"
        return layer_dir / filename

    def _precompute_single_expert(
        self, layer_idx: int, expert_id: int, param_type: ExpertParamType
    ) -> torch.Tensor:
        """Precompute a single expert weight tensor."""

        # Build original parameter name
        param_name = f"block.{layer_idx}.mlp.{param_type.value}"

        # Load full tensor from checkpoint (automatically handles MXFP4)
        full_tensor = self.checkpoint.get(param_name)
        if full_tensor is None:
            raise RuntimeError(f"Failed to load parameter {param_name}")

        # Convert to target dtype
        full_tensor = full_tensor.to(self.target_dtype)

        # Extract expert slice
        if "weight" in param_type.value:
            expert_tensor = full_tensor[expert_id, ...].clone()
        else:  # bias
            expert_tensor = full_tensor[expert_id, :].clone()

        return expert_tensor

    def load_precomputed_expert(self, expert_key: ExpertKey) -> torch.Tensor:
        """Load a precomputed expert tensor by ExpertKey."""
        expert_file = self._get_expert_file_path(
            expert_key.layer_idx, expert_key.expert_id, expert_key.param_type
        )

        if not expert_file.exists():
            raise FileNotFoundError(f"Precomputed expert not found: {expert_file}")

        data = load_file(expert_file, device="cpu")
        return data["tensor"]

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get detailed storage statistics."""

        if not self.precomputed_dir.exists():
            return {"status": "not_precomputed"}

        # Count files and calculate sizes
        total_files = 0
        total_size = 0
        layer_stats = {}

        for layer_idx in range(self.num_layers):
            layer_dir = self.precomputed_dir / f"layer_{layer_idx:02d}"
            layer_files = 0
            layer_size = 0

            if layer_dir.exists():
                expert_files = list(layer_dir.glob("expert_*.safetensors"))
                layer_files = len(expert_files)

                for file_path in expert_files:
                    file_size = file_path.stat().st_size
                    layer_size += file_size
                    total_size += file_size

            layer_stats[f"layer_{layer_idx:02d}"] = {
                "files": layer_files,
                "expected_files": self.num_experts * len(ExpertParamType),
                "size_mb": layer_size / (1024 * 1024),
            }

            total_files += layer_files

        return {
            "status": "precomputed",
            "total_files": total_files,
            "expected_files": self.total_expert_files,
            "completion_rate": (
                total_files / self.total_expert_files
                if self.total_expert_files > 0
                else 0
            ),
            "total_size_mb": total_size / (1024 * 1024),
            "avg_file_size_kb": (
                (total_size / total_files / 1024) if total_files > 0 else 0
            ),
            "precomputed_dir": str(self.precomputed_dir),
            "layer_stats": layer_stats,
        }

    def validate_random_samples(self, num_samples: int = 20) -> bool:
        """Validate precomputed results by comparing with original MXFP4 decoding."""

        print(f"Validating {num_samples} random expert weights...")

        import random

        random.seed(42)  # Reproducible validation

        for i in range(num_samples):
            # Random selection
            layer_idx = random.randint(0, self.num_layers - 1)
            expert_id = random.randint(0, self.num_experts - 1)
            param_type = random.choice(list(ExpertParamType))

            try:
                # Load precomputed result
                expert_key = ExpertKey(
                    layer_idx=layer_idx, expert_id=expert_id, param_type=param_type
                )
                precomputed_tensor = self.load_precomputed_expert(expert_key)

                # Recompute original result
                original_tensor = self._precompute_single_expert(
                    layer_idx, expert_id, param_type
                )

                # Numerical comparison
                diff = torch.abs(precomputed_tensor - original_tensor).max()
                if diff > 1e-6:
                    print(
                        f"❌ Validation failed [{i+1}/{num_samples}]: max diff {diff}"
                    )
                    print(
                        f"   Location: layer_{layer_idx} expert_{expert_id} {param_type.value}"
                    )
                    return False
                else:
                    print(f"✅ Validation passed [{i+1}/{num_samples}]: {expert_key}")

            except Exception as e:
                print(f"❌ Validation error [{i+1}/{num_samples}]: {e}")
                return False

        print("✅ All random validations passed!")
        return True

    def check_precomputed_availability(self) -> bool:
        """Check if precomputed weights are complete and available."""

        metadata_file = self.precomputed_dir / "metadata.json"
        if not metadata_file.exists():
            return False

        try:
            with open(metadata_file) as f:
                metadata = json.load(f)

            # Verify metadata
            if metadata.get("model_type") != self.model_type.value:
                return False

            expected_layers = metadata.get("num_layers", 24)
            expected_experts = metadata.get("num_experts", 32)

            # Check all required files exist
            for layer_idx in range(expected_layers):
                for expert_id in range(expected_experts):
                    for param_type in ExpertParamType:
                        expert_file = self._get_expert_file_path(
                            layer_idx, expert_id, param_type
                        )
                        if not expert_file.exists():
                            return False

            return True

        except Exception as e:
            print(f"Warning: Precomputed availability check failed: {e}")
            return False


# Convenience functions
def precompute_gpt_oss_20b(
    target_dtype: torch.dtype = torch.bfloat16, force_rebuild: bool = False
) -> GPTOSSPrecomputer:
    """Precompute GPT-OSS-20B weights (3072 files)."""
    precomputer = GPTOSSPrecomputer(ModelType.GPT_OSS_20B, target_dtype)
    precomputer.precompute_all_experts(force_rebuild)
    return precomputer


def validate_precomputed_gpt_oss_20b(num_samples: int = 20) -> bool:
    """Validate precomputed GPT-OSS-20B results."""
    precomputer = GPTOSSPrecomputer(ModelType.GPT_OSS_20B)
    return precomputer.validate_random_samples(num_samples)


def get_precomputed_storage_stats() -> Dict[str, Any]:
    """Get precomputed storage statistics."""
    precomputer = GPTOSSPrecomputer(ModelType.GPT_OSS_20B)
    return precomputer.get_storage_stats()


def is_precomputed_available() -> bool:
    """Check if precomputed weights are available."""
    precomputer = GPTOSSPrecomputer(ModelType.GPT_OSS_20B)
    return precomputer.check_precomputed_availability()


# Example usage
if __name__ == "__main__":
    # Precompute all weights
    print("Starting precomputation...")
    precomputer = precompute_gpt_oss_20b()

    # Show statistics
    stats = get_precomputed_storage_stats()
    print(f"\nStorage stats: {stats}")

    # Validate results
    print("\nValidating precomputed weights...")
    is_valid = validate_precomputed_gpt_oss_20b()
    print(f"Validation result: {'PASSED' if is_valid else 'FAILED'}")
