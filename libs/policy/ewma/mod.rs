//! EWMA (Exponentially Weighted Moving Average) predictor module
//!
//! This module implements the layer-local EWMA algorithm for predicting expert activation
//! probabilities in Mixture-of-Experts models. It provides:
//!
//! - predictor::EwmaPredictor: Main predictor with shared Timer integration
//! - config::EwmaConfig: Configuration for EWMA parameters (α, initialization value)  
//! - error::EwmaError: Error handling for EWMA operations
//!
//! The implementation uses 0-based indexing consistent with our implementation convention,
//! mathematically equivalent to the 1-based formulas in the documentation.

pub mod config;
pub mod error;
pub mod predictor;
