//! Probability fusion module
//!
//! This module implements the probability fusion algorithm that combines EWMA and ScoutGate
//! predictions with forward-causal weighting based on layer reuse distances.
//!
//! ## Algorithm Overview
//!
//! The fusion process follows these steps:
//!
//! 1. **Base Fusion**: Blend EWMA and ScoutGate predictions
//!    ```
//!    p^{base}_{e,ℓ}(t) := (1-η)·p̂^{EWMA}_{e,ℓ}(t) + η·p̂^{SG}_{e,ℓ}(t)
//!    ```
//!
//! 2. **Reuse Distance**: Calculate layer-step distance until target layer is needed again
//!    ```
//!    D(ℓ|ℓ(t)) = {
//!      ℓ - ℓ(t),           if ℓ >= ℓ(t) (future layers in current token)
//!      (L - ℓ(t)) + ℓ,     if ℓ < ℓ(t) (next token layers)
//!    }
//!    ```
//!
//! 3. **Forward-Causal Weights**: Apply exponential decay based on reuse distance  
//!    ```
//!    W(ℓ|ℓ(t)) := e^{-γ·D(ℓ|ℓ(t))}
//!    ```
//!
//! 4. **Final Fusion**: Combine base prediction with causal weight
//!    ```
//!    p^{fuse}_{e,ℓ}(t) := p^{base}_{e,ℓ}(t) · W(ℓ|ℓ(t))
//!    ```
//!
//! ## Usage Example
//!
//! ```rust
//! use policy::fusion::ProbabilityFusion;
//! use policy::ExpertKey;
//! use std::collections::HashMap;
//!
//! // Create fusion predictor
//! let fusion = ProbabilityFusion::for_gptoss20b();
//!
//! // Prepare prediction maps
//! let mut ewma_preds = HashMap::new();
//! let mut scoutgate_preds = HashMap::new();
//!
//! let expert_key = ExpertKey::new(0, 1); // expert_id=0, layer_id=1
//! ewma_preds.insert(expert_key, 0.7);
//! scoutgate_preds.insert(expert_key, 0.3);
//!
//! // Perform fusion for current layer 0
//! let fused = fusion.fuse_predictions(&ewma_preds, &scoutgate_preds, 0)?;
//! let fused_prob = fused[&expert_key];
//! ```

pub mod error;
pub mod predictor;

pub use error::FusionError;
pub use predictor::ProbabilityFusion;
