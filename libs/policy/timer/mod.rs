/// Simple timer for MoE layer execution tracking
/// Maintains global time and provides layer-related calculations
/// Uses 0-based indexing: time starts at 0, layers are 0..L-1
pub struct Timer {
    current_time: u64,   // Current global time t (starts from 0)
    total_layers: usize, // Total number of layers L
}

/// Timer error type - covers all invalid input scenarios
#[derive(Debug, Clone, PartialEq)]
pub enum TimerError {
    Invalid, // Covers all invalid input cases (layer>=L, L=0, etc.)
}

impl std::fmt::Display for TimerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimerError::Invalid => write!(f, "Invalid timer parameters"),
        }
    }
}

impl std::error::Error for TimerError {}

impl Timer {
    /// Create a new timer instance
    ///
    /// # Arguments
    /// * `total_layers` - Total number of transformer layers (must be > 0)
    pub fn new(total_layers: usize) -> Result<Self, TimerError> {
        if total_layers == 0 {
            return Err(TimerError::Invalid);
        }
        Ok(Timer {
            current_time: 0,
            total_layers,
        })
    }

    /// Create a timer instance configured for GPT-OSS-20B model
    /// Uses the predefined configuration: 24 layers (block.0 to block.23)
    pub fn from_gptoss20b() -> Self {
        use crate::constants::models::GPT_OSS_20B;
        Timer {
            current_time: 0,
            total_layers: GPT_OSS_20B.total_layers,
        }
    }

    /// Create a timer instance configured for GPT-OSS-120B model
    /// Uses the predefined configuration: 36 layers
    pub fn from_gptoss120b() -> Self {
        use crate::constants::models::GPT_OSS_120B;
        Timer {
            current_time: 0,
            total_layers: GPT_OSS_120B.total_layers,
        }
    }

    /// Create a timer instance configured for Phi-Tiny-MoE model
    /// Uses the predefined configuration: 8 layers (for testing)
    pub fn from_phi_tiny_moe() -> Self {
        use crate::constants::models::PHI_TINY_MOE;
        Timer {
            current_time: 0,
            total_layers: PHI_TINY_MOE.total_layers,
        }
    }

    /// Calculate current executing layer from global time
    /// Formula: ℓ(t) = t mod L (0-based indexing)
    ///
    /// # Arguments
    /// * `time` - Global time t (starts from 0)
    /// * `total_layers` - Total number of layers L (must be > 0)
    pub fn get_current_layer(time: u64, total_layers: usize) -> Result<usize, TimerError> {
        if total_layers == 0 {
            return Err(TimerError::Invalid);
        }
        Ok((time % (total_layers as u64)) as usize)
    }

    /// Calculate visit count for a specific layer up to given time
    /// Formula: v_ℓ(t) = ⌊t/L⌋ + (1 if t%L >= ℓ else 0)
    ///
    /// # Arguments
    /// * `layer` - Target layer (0-based, must be < total_layers)
    /// * `time` - Global time t (starts from 0)
    /// * `total_layers` - Total number of layers L (must be > 0)
    pub fn get_visit_count(
        layer: usize,
        time: u64,
        total_layers: usize,
    ) -> Result<u64, TimerError> {
        if layer >= total_layers || total_layers == 0 {
            return Err(TimerError::Invalid);
        }

        let full_cycles = time / (total_layers as u64);
        let current_position = (time % (total_layers as u64)) as usize;
        let extra = if current_position >= layer { 1 } else { 0 };

        Ok(full_cycles + extra)
    }

    /// Advance timer by one step
    pub fn tick(&mut self) {
        self.current_time += 1;
    }

    /// Get current global time
    pub fn current_time(&self) -> u64 {
        self.current_time
    }

    /// Get currently executing layer
    pub fn current_layer(&self) -> Result<usize, TimerError> {
        Self::get_current_layer(self.current_time, self.total_layers)
    }

    /// Get total number of layers
    pub fn total_layers(&self) -> usize {
        self.total_layers
    }

    /// Get visit count for a specific layer at current time
    pub fn layer_visit_count(&self, layer: usize) -> Result<u64, TimerError> {
        Self::get_visit_count(layer, self.current_time, self.total_layers)
    }
}
