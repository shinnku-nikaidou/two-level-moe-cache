// Policy library module
pub mod constants;
pub mod ewma;
pub mod fusion;
pub mod scoutgate;
pub mod timer;
pub mod watermark;

/// Expert parameter type for MoE expert weights
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertParamType {
    MLP1Weight, // First MLP layer weights (up projection)
    MLP1Bias,   // First MLP layer bias
    MLP2Weight, // Second MLP layer weights (down projection)
    MLP2Bias,   // Second MLP layer bias
}

impl ExpertParamType {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            ExpertParamType::MLP1Weight => "mlp1_weight",
            ExpertParamType::MLP1Bias => "mlp1_bias",
            ExpertParamType::MLP2Weight => "mlp2_weight",
            ExpertParamType::MLP2Bias => "mlp2_bias",
        }
    }

    /// Get all parameter types for an expert
    pub fn all() -> Vec<Self> {
        vec![
            ExpertParamType::MLP1Weight,
            ExpertParamType::MLP1Bias,
            ExpertParamType::MLP2Weight,
            ExpertParamType::MLP2Bias,
        ]
    }
}

/// Expert-Layer key for indexing MoE experts across different layers
/// Uses 0-based indexing consistent with our implementation convention
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub expert_id: usize,            // Expert index within the layer (0-based)
    pub layer_id: usize,             // Layer index in the model (0-based)
    pub param_type: ExpertParamType, // Parameter type within the expert
}

impl ExpertKey {
    /// Create a new ExpertKey
    pub fn new(expert_id: usize, layer_id: usize, param_type: ExpertParamType) -> Self {
        Self {
            expert_id,
            layer_id,
            param_type,
        }
    }

    /// Create an ExpertKey with validation against model configuration
    pub fn with_validation(
        expert_id: usize,
        layer_id: usize,
        param_type: ExpertParamType,
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

        Ok(Self::new(expert_id, layer_id, param_type))
    }

    /// Get all expert keys for a specific layer (all experts, all param types)
    pub fn layer_experts(layer_id: usize, num_experts: usize) -> Vec<Self> {
        let mut keys = Vec::new();
        for expert_id in 0..num_experts {
            for param_type in ExpertParamType::all() {
                keys.push(Self::new(expert_id, layer_id, param_type));
            }
        }
        keys
    }

    /// Get all expert keys for the entire model (all layers, all experts, all param types)
    pub fn all_experts(config: &constants::models::ModelConfig) -> Vec<Self> {
        let mut keys = Vec::new();
        for layer_id in 0..config.total_layers {
            for expert_id in 0..config.experts_per_layer {
                for param_type in ExpertParamType::all() {
                    keys.push(Self::new(expert_id, layer_id, param_type));
                }
            }
        }
        keys
    }

    /// Get all parameter keys for a specific expert in a layer
    pub fn expert_params(expert_id: usize, layer_id: usize) -> Vec<Self> {
        ExpertParamType::all()
            .into_iter()
            .map(|param_type| Self::new(expert_id, layer_id, param_type))
            .collect()
    }

    /// Create a simple key for expert-level operations (uses MLP1Weight as default param)
    pub fn expert_level(expert_id: usize, layer_id: usize) -> Self {
        Self::new(expert_id, layer_id, ExpertParamType::MLP1Weight)
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
