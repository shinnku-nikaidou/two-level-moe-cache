// EWMA parameters
pub const ALPHA: f64 = 0.3;

// Model configurations
/// Model configuration structure
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub name: &'static str,
    pub total_layers: usize,
    pub experts_per_layer: usize,
}

/// Python-compatible model type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    GptOss20B,
    GptOss120B,
    PhiTinyMoe,
}

impl From<ModelType> for ModelConfig {
    fn from(model_type: ModelType) -> Self {
        match model_type {
            ModelType::GptOss20B => GPT_OSS_20B,
            ModelType::GptOss120B => GPT_OSS_120B,
            ModelType::PhiTinyMoe => PHI_TINY_MOE,
        }
    }
}

// GPT-OSS model series configurations
pub const GPT_OSS_20B: ModelConfig = ModelConfig {
    name: "gpt-oss-20b",
    total_layers: 24,     // block.0 to block.23
    experts_per_layer: 8, // Typical MoE configuration
};

pub const GPT_OSS_120B: ModelConfig = ModelConfig {
    name: "gpt-oss-120b",
    total_layers: 36,      // Assumed for 120B model
    experts_per_layer: 16, // More experts for larger model
};

// Phi-Tiny-MoE configuration (for testing)
pub const PHI_TINY_MOE: ModelConfig = ModelConfig {
    name: "phi-tiny-moe",
    total_layers: 8, // Small model for testing
    experts_per_layer: 4,
};

/// Get model configuration by name
pub fn get_model_config(name: &str) -> Option<&'static ModelConfig> {
    match name {
        "gpt-oss-20b" => Some(&GPT_OSS_20B),
        "gpt-oss-120b" => Some(&GPT_OSS_120B),
        "phi-tiny-moe" => Some(&PHI_TINY_MOE),
        _ => None,
    }
}

/// List all available model configurations
pub fn available_models() -> Vec<&'static ModelConfig> {
    vec![&GPT_OSS_20B, &GPT_OSS_120B, &PHI_TINY_MOE]
}
