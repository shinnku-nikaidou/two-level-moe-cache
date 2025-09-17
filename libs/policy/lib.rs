// Policy library module
pub mod constants;
pub mod ewma;
pub mod fusion;
pub mod scoutgate;
pub mod timer;
pub mod watermark;

/// Abstract expert identifier for policy-level operations
/// Unlike ExpertKey, this doesn't include parameter types - policy layer
/// only cares about expert-level decisions, not individual parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AbstractExpert {
    pub expert_id: usize, // Expert index within the layer (0-based)
    pub layer_id: usize,  // Layer index in the model (0-based)
}

impl AbstractExpert {
    /// Create a new AbstractExpert
    pub fn new(expert_id: usize, layer_id: usize) -> Self {
        Self {
            expert_id,
            layer_id,
        }
    }

    /// Create an AbstractExpert with validation against model configuration
    pub fn new_with_validation(
        expert_id: usize,
        layer_id: usize,
        config: &constants::models::ModelConfig,
    ) -> Result<Self, AbstractExpertError> {
        if layer_id >= config.total_layers {
            return Err(AbstractExpertError::InvalidLayer {
                layer_id,
                max_layers: config.total_layers,
            });
        }

        if expert_id >= config.experts_per_layer {
            return Err(AbstractExpertError::InvalidExpert {
                expert_id,
                max_experts: config.experts_per_layer,
            });
        }

        Ok(Self::new(expert_id, layer_id))
    }

    /// Get all abstract experts for a specific layer
    pub fn layer_experts(layer_id: usize, num_experts: usize) -> Vec<Self> {
        (0..num_experts)
            .map(|expert_id| Self::new(expert_id, layer_id))
            .collect()
    }

    /// Get all abstract experts for the entire model
    pub fn all_experts(config: &constants::models::ModelConfig) -> Vec<Self> {
        let mut experts = Vec::new();
        for layer_id in 0..config.total_layers {
            for expert_id in 0..config.experts_per_layer {
                experts.push(Self::new(expert_id, layer_id));
            }
        }
        experts
    }
}

/// Error types for AbstractExpert operations
#[derive(Debug, Clone, PartialEq)]
pub enum AbstractExpertError {
    InvalidLayer {
        layer_id: usize,
        max_layers: usize,
    },
    InvalidExpert {
        expert_id: usize,
        max_experts: usize,
    },
}

impl std::fmt::Display for AbstractExpertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AbstractExpertError::InvalidLayer {
                layer_id,
                max_layers,
            } => {
                write!(
                    f,
                    "Invalid layer_id {} (must be < {})",
                    layer_id, max_layers
                )
            }
            AbstractExpertError::InvalidExpert {
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

impl std::error::Error for AbstractExpertError {}
