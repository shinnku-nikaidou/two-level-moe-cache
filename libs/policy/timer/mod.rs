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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_mapping() {
        // Test ℓ(t) = t mod L (0-based indexing)
        assert_eq!(Timer::get_current_layer(0, 3).unwrap(), 0);
        assert_eq!(Timer::get_current_layer(1, 3).unwrap(), 1);
        assert_eq!(Timer::get_current_layer(2, 3).unwrap(), 2);
        assert_eq!(Timer::get_current_layer(3, 3).unwrap(), 0);
        assert_eq!(Timer::get_current_layer(4, 3).unwrap(), 1);
        assert_eq!(Timer::get_current_layer(5, 3).unwrap(), 2);
    }

    #[test]
    fn test_visit_count() {
        // Test v_ℓ(t) = ⌊t/L⌋ + (1 if t%L >= ℓ else 0)
        // For 3 layers: [0, 1, 2, 0, 1, 2, 0, 1, 2, ...]

        // Layer 0 visits
        assert_eq!(Timer::get_visit_count(0, 0, 3).unwrap(), 1); // t=0: layer 0 executing
        assert_eq!(Timer::get_visit_count(0, 1, 3).unwrap(), 1); // t=1: layer 1 executing, layer 0 visited once
        assert_eq!(Timer::get_visit_count(0, 2, 3).unwrap(), 1); // t=2: layer 2 executing, layer 0 visited once
        assert_eq!(Timer::get_visit_count(0, 3, 3).unwrap(), 2); // t=3: layer 0 executing again

        // Layer 1 visits
        assert_eq!(Timer::get_visit_count(1, 0, 3).unwrap(), 0); // t=0: layer 1 not visited yet
        assert_eq!(Timer::get_visit_count(1, 1, 3).unwrap(), 1); // t=1: layer 1 executing
        assert_eq!(Timer::get_visit_count(1, 4, 3).unwrap(), 2); // t=4: layer 1 executing second time

        // Layer 2 visits
        assert_eq!(Timer::get_visit_count(2, 0, 3).unwrap(), 0); // t=0: layer 2 not visited yet
        assert_eq!(Timer::get_visit_count(2, 1, 3).unwrap(), 0); // t=1: layer 2 not visited yet
        assert_eq!(Timer::get_visit_count(2, 2, 3).unwrap(), 1); // t=2: layer 2 executing
        assert_eq!(Timer::get_visit_count(2, 5, 3).unwrap(), 2); // t=5: layer 2 executing second time
    }

    #[test]
    fn test_edge_cases() {
        // Boundary conditions
        assert!(Timer::get_current_layer(0, 0).is_err()); // No layers
        assert_eq!(Timer::get_current_layer(10, 1).unwrap(), 0); // Single layer

        // Invalid layer indices
        assert!(Timer::get_visit_count(3, 0, 3).is_err()); // layer >= total_layers
        assert!(Timer::get_visit_count(0, 0, 0).is_err()); // No layers
    }

    #[test]
    fn test_timer_instance() {
        let mut timer = Timer::new(3).unwrap();

        // Initial state (t=0)
        assert_eq!(timer.current_time(), 0);
        assert_eq!(timer.current_layer().unwrap(), 0); // Layer 0 at t=0
        assert_eq!(timer.layer_visit_count(0).unwrap(), 1); // Layer 0 is currently executing
        assert_eq!(timer.layer_visit_count(1).unwrap(), 0); // Layer 1 not visited yet
        assert_eq!(timer.layer_visit_count(2).unwrap(), 0); // Layer 2 not visited yet

        // First tick (t=1)
        timer.tick();
        assert_eq!(timer.current_time(), 1);
        assert_eq!(timer.current_layer().unwrap(), 1); // Layer 1 at t=1
        assert_eq!(timer.layer_visit_count(0).unwrap(), 1); // Layer 0 visited once
        assert_eq!(timer.layer_visit_count(1).unwrap(), 1); // Layer 1 currently executing
        assert_eq!(timer.layer_visit_count(2).unwrap(), 0); // Layer 2 not visited yet

        // Second tick (t=2)
        timer.tick();
        assert_eq!(timer.current_layer().unwrap(), 2); // Layer 2 at t=2
        assert_eq!(timer.layer_visit_count(2).unwrap(), 1); // Layer 2 currently executing

        // Third tick (t=3) - cycles back to layer 0
        timer.tick();
        assert_eq!(timer.current_layer().unwrap(), 0); // Layer 0 at t=3
        assert_eq!(timer.layer_visit_count(0).unwrap(), 2); // Layer 0 visited twice now
    }

    #[test]
    fn test_invalid_creation() {
        assert!(Timer::new(0).is_err());
    }

    #[test]
    fn test_visit_count_comprehensive() {
        // Test comprehensive visit counting for different scenarios

        // Single layer case
        assert_eq!(Timer::get_visit_count(0, 0, 1).unwrap(), 1);
        assert_eq!(Timer::get_visit_count(0, 5, 1).unwrap(), 6); // Every time step is layer 0

        // Two layer case: [0, 1, 0, 1, 0, 1, ...]
        assert_eq!(Timer::get_visit_count(0, 0, 2).unwrap(), 1); // t=0: executing layer 0
        assert_eq!(Timer::get_visit_count(0, 1, 2).unwrap(), 1); // t=1: layer 0 visited once
        assert_eq!(Timer::get_visit_count(0, 2, 2).unwrap(), 2); // t=2: executing layer 0 again
        assert_eq!(Timer::get_visit_count(1, 1, 2).unwrap(), 1); // t=1: executing layer 1
        assert_eq!(Timer::get_visit_count(1, 3, 2).unwrap(), 2); // t=3: executing layer 1 again
    }
}
