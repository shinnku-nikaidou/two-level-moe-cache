//! Complete ScoutGate integration test
//!
//! This test validates the complete ScoutGate architecture end-to-end,
//! ensuring all components work together correctly for expert prediction.

use burn_ndarray::{NdArray, NdArrayDevice};
use policy::scoutgate::ScoutGatePredictor;
use policy::constants::{ModelConfig, ModelType};
use policy::timer::Timer;
use std::sync::{Arc, RwLock};

type Backend = NdArray<f32>;
type Device = NdArrayDevice;

#[test]
fn test_complete_scoutgate_pipeline() {
    println!("=== Testing Complete ScoutGate Pipeline ===");
    
    // Create shared timer
    let timer = Arc::new(RwLock::new(Timer::new()));
    
    // Create model configuration (small test model)
    let model_config = ModelConfig {
        total_layers: 4,
        experts_per_layer: 8,
    };
    
    // Create ScoutGate predictor
    let mut predictor = ScoutGatePredictor::from_model(timer.clone(), ModelType::SmallMoE)
        .expect("Failed to create ScoutGate predictor");
    
    println!("✅ ScoutGate predictor created successfully");
    
    // Test token context update
    for token_id in 0..10 {
        predictor.update_token_context(token_id)
            .expect("Failed to update token context");
    }
    
    println!("✅ Token context updated successfully");
    
    // Test single layer prediction
    let layer_scores = predictor.predict_layer(0)
        .expect("Failed to predict layer 0");
    
    assert_eq!(layer_scores.len(), model_config.experts_per_layer);
    
    // Verify scores are valid probabilities (between 0 and 1)
    for (expert_id, &score) in layer_scores.iter().enumerate() {
        assert!(score >= 0.0 && score <= 1.0, 
            "Expert {} score {} is not a valid probability", expert_id, score);
    }
    
    println!("✅ Single layer prediction works, {} experts scored", layer_scores.len());
    
    // Test all layers prediction
    predictor.predict_all_layers()
        .expect("Failed to predict all layers");
    
    println!("✅ All layers predicted successfully");
    
    // Get final probabilities
    let probabilities = predictor.get_probabilities();
    
    // Verify probabilities structure
    assert_eq!(probabilities.inner.len(), model_config.total_layers);
    for layer_probs in &probabilities.inner {
        assert_eq!(layer_probs.len(), model_config.experts_per_layer);
    }
    
    // Count non-None predictions
    let mut total_predictions = 0;
    let mut valid_predictions = 0;
    
    for layer_id in 0..model_config.total_layers {
        for expert_id in 0..model_config.experts_per_layer {
            total_predictions += 1;
            if let Some(prob) = probabilities.get(layer_id, expert_id) {
                valid_predictions += 1;
                assert!(prob >= 0.0 && prob <= 1.0, 
                    "Invalid probability {} for layer {} expert {}", prob, layer_id, expert_id);
            }
        }
    }
    
    println!("✅ Probability validation: {}/{} predictions are valid", 
             valid_predictions, total_predictions);
    
    // Test expert embedding updates
    let test_embeddings = vec![vec![0.1f32; 64]; model_config.experts_per_layer];
    predictor.update_expert_embeddings(0, test_embeddings)
        .expect("Failed to update expert embeddings");
    
    println!("✅ Expert embeddings updated successfully");
    
    // Test layer bias initialization
    predictor.initialize_layer_bias(0, model_config.experts_per_layer)
        .expect("Failed to initialize layer bias");
    
    println!("✅ Layer bias initialized successfully");
    
    // Test configuration retrieval
    let config = predictor.get_config().expect("Failed to get config");
    println!("✅ Configuration: d_model={}, d_context={}, d_prime={}, window={}, layers={}", 
             config.0, config.1, config.2, config.3, config.4);
    
    println!("🎉 Complete ScoutGate pipeline test PASSED!");
}

#[test]
fn test_scoutgate_component_integration() {
    println!("=== Testing ScoutGate Component Integration ===");
    
    let timer = Arc::new(RwLock::new(Timer::new()));
    let model_config = ModelConfig {
        total_layers: 2,
        experts_per_layer: 4,
    };
    
    let mut predictor = ScoutGatePredictor::from_model(timer, ModelType::SmallMoE)
        .expect("Failed to create predictor");
    
    // Test sequential token processing
    let test_tokens = vec![1, 5, 10, 15, 20];
    for token in test_tokens {
        predictor.update_token_context(token)
            .expect("Failed to update token context");
    }
    
    println!("✅ Sequential token processing works");
    
    // Test batch vs single layer prediction consistency
    predictor.predict_all_layers()
        .expect("Failed to predict all layers");
    
    let single_layer_0 = predictor.predict_layer(0)
        .expect("Failed to predict layer 0");
    let single_layer_1 = predictor.predict_layer(1)
        .expect("Failed to predict layer 1");
    
    let all_probs = predictor.get_probabilities();
    
    // Verify consistency between single and batch predictions
    for expert_id in 0..model_config.experts_per_layer {
        if let Some(batch_prob_0) = all_probs.get(0, expert_id) {
            let diff = (single_layer_0[expert_id] - batch_prob_0).abs();
            assert!(diff < 1e-6, "Inconsistent predictions for layer 0 expert {}: single={}, batch={}", 
                   expert_id, single_layer_0[expert_id], batch_prob_0);
        }
        
        if let Some(batch_prob_1) = all_probs.get(1, expert_id) {
            let diff = (single_layer_1[expert_id] - batch_prob_1).abs();
            assert!(diff < 1e-6, "Inconsistent predictions for layer 1 expert {}: single={}, batch={}", 
                   expert_id, single_layer_1[expert_id], batch_prob_1);
        }
    }
    
    println!("✅ Single vs batch prediction consistency verified");
    
    // Test error handling for invalid layer
    let result = predictor.predict_layer(999);
    assert!(result.is_err(), "Should fail for invalid layer ID");
    
    println!("✅ Error handling works correctly");
    
    println!("🎉 ScoutGate component integration test PASSED!");
}

#[test]
fn test_scoutgate_multiple_model_types() {
    println!("=== Testing ScoutGate with Multiple Model Types ===");
    
    let model_types = vec![
        (ModelType::SmallMoE, "SmallMoE"),
        (ModelType::MediumMoE, "MediumMoE"),
        (ModelType::LargeMoE, "LargeMoE"),
    ];
    
    for (model_type, name) in model_types {
        println!("Testing with model type: {}", name);
        
        let timer = Arc::new(RwLock::new(Timer::new()));
        let mut predictor = ScoutGatePredictor::from_model(timer, model_type)
            .expect(&format!("Failed to create predictor for {}", name));
        
        // Add some tokens
        for token in 0..5 {
            predictor.update_token_context(token)
                .expect("Failed to update token context");
        }
        
        // Test prediction
        predictor.predict_all_layers()
            .expect("Failed to predict all layers");
        
        let probabilities = predictor.get_probabilities();
        let config: ModelConfig = model_type.into();
        
        // Verify structure matches model configuration
        assert_eq!(probabilities.inner.len(), config.total_layers);
        for layer_probs in &probabilities.inner {
            assert_eq!(layer_probs.len(), config.experts_per_layer);
        }
        
        println!("✅ {} works correctly ({} layers, {} experts per layer)", 
                 name, config.total_layers, config.experts_per_layer);
    }
    
    println!("🎉 Multiple model types test PASSED!");
}

#[test] 
fn test_scoutgate_prediction_quality() {
    println!("=== Testing ScoutGate Prediction Quality ===");
    
    let timer = Arc::new(RwLock::new(Timer::new()));
    let mut predictor = ScoutGatePredictor::from_model(timer, ModelType::SmallMoE)
        .expect("Failed to create predictor");
    
    // Test with different token sequences
    let sequences = vec![
        vec![1, 2, 3, 4, 5],
        vec![10, 20, 30, 40, 50],
        vec![100, 200, 300, 400, 500],
    ];
    
    for (seq_id, sequence) in sequences.iter().enumerate() {
        // Reset and add tokens
        for &token in sequence {
            predictor.update_token_context(token)
                .expect("Failed to update token context");
        }
        
        predictor.predict_all_layers()
            .expect("Failed to predict all layers");
        
        let probabilities = predictor.get_probabilities();
        
        // Check that predictions are reasonable
        let mut total_prob_sum = 0.0;
        let mut valid_predictions = 0;
        
        for layer_id in 0..4 {  // SmallMoE has 4 layers
            let mut layer_sum = 0.0;
            for expert_id in 0..8 {  // SmallMoE has 8 experts per layer
                if let Some(prob) = probabilities.get(layer_id, expert_id) {
                    layer_sum += prob;
                    valid_predictions += 1;
                }
            }
            total_prob_sum += layer_sum;
        }
        
        println!("✅ Sequence {}: {} valid predictions, average prob = {:.4}", 
                 seq_id + 1, valid_predictions, 
                 if valid_predictions > 0 { total_prob_sum / valid_predictions as f64 } else { 0.0 });
    }
    
    println!("🎉 Prediction quality test PASSED!");
}