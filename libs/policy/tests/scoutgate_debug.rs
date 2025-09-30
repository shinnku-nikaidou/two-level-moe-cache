//! Simple ScoutGate debug test
//!
//! This test debugs the ScoutGate tensor dimension issues

use policy::scoutgate::ScoutGatePredictor;
use policy::constants::ModelType;
use policy::timer::Timer;
use std::sync::{Arc, RwLock};

#[test]
fn test_scoutgate_dimensions_debug() {
    println!("=== ScoutGate Dimensions Debug Test ===");
    
    // Create a simple timer and predictor
    let timer = Arc::new(RwLock::new(Timer::new(4))); // Just 4 layers
    let mut predictor = ScoutGatePredictor::from_model(timer.clone(), ModelType::PhiTinyMoe)
        .expect("Failed to create ScoutGate predictor");
    
    println!("✅ ScoutGate predictor created");
    
    // Add just one token to avoid complexity
    predictor.update_token_context(42)
        .expect("Failed to add token");
    
    println!("✅ Token added to context");
    
    // Try to predict just the first layer to see what happens
    match predictor.predict_layer(0) {
        Ok(scores) => {
            println!("✅ Layer 0 prediction successful: {} scores", scores.len());
            for (i, score) in scores.iter().take(5).enumerate() {
                println!("  Expert {}: {:.4}", i, score);
            }
        }
        Err(e) => {
            println!("❌ Layer 0 prediction failed: {:?}", e);
        }
    }
    
    println!("🔍 Debug test completed");
}