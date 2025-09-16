// Policy library module
pub mod constants;
pub mod ewma;
pub mod scoutgate;
pub mod timer;
pub mod watermark;

/// Expert-Layer key for indexing MoE experts across different layers
/// Uses 0-based indexing consistent with our implementation convention
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub expert_id: usize, // Expert index within the layer (0-based)
    pub layer_id: usize,  // Layer index in the model (0-based)
}

impl ExpertKey {
    /// Create a new ExpertKey
    pub fn new(expert_id: usize, layer_id: usize) -> Self {
        Self {
            expert_id,
            layer_id,
        }
    }

    /// Create an ExpertKey with validation against model configuration
    pub fn with_validation(
        expert_id: usize,
        layer_id: usize,
        config: &constants::models::ModelConfig,
    ) -> Result<Self, ExpertKeyError> {
        if layer_id >= config.total_layers {
            return Err(ExpertKeyError::InvalidLayer {
                layer_id,
                max_layers: config.total_layers,
            });
        }

        if expert_id >= config.experts_per_layer {
            return Err(ExpertKeyError::InvalidExpert {
                expert_id,
                max_experts: config.experts_per_layer,
            });
        }

        Ok(Self::new(expert_id, layer_id))
    }

    /// Get all expert keys for a specific layer
    pub fn layer_experts(layer_id: usize, num_experts: usize) -> Vec<Self> {
        (0..num_experts)
            .map(|expert_id| Self::new(expert_id, layer_id))
            .collect()
    }

    /// Get all expert keys for the entire model
    pub fn all_experts(config: &constants::models::ModelConfig) -> Vec<Self> {
        let mut keys = Vec::new();
        for layer_id in 0..config.total_layers {
            for expert_id in 0..config.experts_per_layer {
                keys.push(Self::new(expert_id, layer_id));
            }
        }
        keys
    }
}

/// Error types for ExpertKey operations
#[derive(Debug, Clone, PartialEq)]
pub enum ExpertKeyError {
    InvalidLayer {
        layer_id: usize,
        max_layers: usize,
    },
    InvalidExpert {
        expert_id: usize,
        max_experts: usize,
    },
}

impl std::fmt::Display for ExpertKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpertKeyError::InvalidLayer {
                layer_id,
                max_layers,
            } => {
                write!(
                    f,
                    "Invalid layer_id {} (must be < {})",
                    layer_id, max_layers
                )
            }
            ExpertKeyError::InvalidExpert {
                expert_id,
                max_experts,
            } => {
                write!(
                    f,
                    "Invalid expert_id {} (must be < {})",
                    expert_id, max_experts
                )
            }
        }
    }
}

impl std::error::Error for ExpertKeyError {}
