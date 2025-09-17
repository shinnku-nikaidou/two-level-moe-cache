/// Simple timer for MoE layer execution tracking
/// Maintains global time and provides layer-related calculations
/// Uses 0-based indexing: time starts at 0, layers are 0..L-1
pub struct Timer {
    current_time: u64,   // Current global time t (starts from 0)
    total_layers: usize, // Total number of layers L
}

impl Timer {
    /// Create a new timer instance
    ///
    /// # Arguments
    /// * `total_layers` - Total number of transformer layers (must be > 0)
    pub fn new(total_layers: usize) -> Self {
        assert!(total_layers > 0, "Total layers must be greater than 0");
        Timer {
            current_time: 0,
            total_layers,
        }
    }

    /// Create a timer instance from model configuration
    ///
    /// # Arguments
    /// * `config` - Model configuration containing layer information
    pub fn from_model(config: &crate::constants::ModelConfig) -> Self {
        use crate::constants::{GPT_OSS_20B, GPT_OSS_120B, PHI_TINY_MOE};

        match config.name {
            "gpt-oss-20b" => Timer {
                current_time: 0,
                total_layers: GPT_OSS_20B.total_layers,
            },
            "gpt-oss-120b" => Timer {
                current_time: 0,
                total_layers: GPT_OSS_120B.total_layers,
            },
            "phi-tiny-moe" => Timer {
                current_time: 0,
                total_layers: PHI_TINY_MOE.total_layers,
            },
            _ => panic!("Unsupported model type: {}", config.name),
        }
    }

    /// Calculate current executing layer from global time
    /// Formula: ℓ(t) = t mod L (0-based indexing)
    pub fn current_layer(&self) -> usize {
        (self.current_time % (self.total_layers as u64)) as usize
    }

    /// Calculate visit count for a specific layer up to current time
    /// Formula: v_ℓ(t) = ⌊t/L⌋ + (1 if t%L >= ℓ else 0)
    pub fn layer_visit_count(&self, layer: usize) -> u64 {
        assert!(layer < self.total_layers, "Layer index out of bounds");

        let full_cycles = self.current_time / (self.total_layers as u64);
        let current_position = (self.current_time % (self.total_layers as u64)) as usize;
        let extra = if current_position >= layer { 1 } else { 0 };

        full_cycles + extra
    }

    /// Advance timer by one step
    pub fn step(&mut self) {
        self.current_time += 1;
    }

    /// Get current global time
    pub fn current_time(&self) -> u64 {
        self.current_time
    }

    /// Get total number of layers
    pub fn total_layers(&self) -> usize {
        self.total_layers
    }
}
