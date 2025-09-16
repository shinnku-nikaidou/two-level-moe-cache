use std::rc::Rc;

use policy::ExpertKey;
use policy::ewma::config::EwmaConfig;
use policy::ewma::error::EwmaError;
use policy::ewma::predictor::EwmaPredictor;
use policy::timer::Timer;

fn create_test_timer() -> Rc<Timer> {
    Rc::new(Timer::new(4).unwrap()) // 4 layers for testing
}

#[test]
fn test_ewma_creation() {
    let timer = create_test_timer();
    let config = policy::constants::models::PHI_TINY_MOE.clone();
    let ewma_config = EwmaConfig::default();

    let predictor = EwmaPredictor::new(timer, config, ewma_config);
    assert!(predictor.is_ok());

    let predictor = predictor.unwrap();
    assert_eq!(predictor.alpha(), policy::constants::ALPHA);
    assert_eq!(predictor.num_tracked_experts(), 0);
}

#[test]
fn test_invalid_alpha() {
    let timer = create_test_timer();
    let config = policy::constants::models::PHI_TINY_MOE.clone();

    // Test alpha = 0 (invalid)
    let ewma_config_zero = EwmaConfig {
        alpha: 0.0,
        ..EwmaConfig::default()
    };
    let result = EwmaPredictor::new(timer.clone(), config.clone(), ewma_config_zero);
    assert!(matches!(result, Err(EwmaError::InvalidAlpha(_))));

    // Test alpha > 1 (invalid)
    let ewma_config_large = EwmaConfig {
        alpha: 1.5,
        ..EwmaConfig::default()
    };
    let result = EwmaPredictor::new(timer, config, ewma_config_large);
    assert!(matches!(result, Err(EwmaError::InvalidAlpha(_))));
}

#[test]
fn test_invalid_init_value() {
    let timer = create_test_timer();
    let config = policy::constants::models::PHI_TINY_MOE.clone();

    // Test negative initialization value (invalid)
    let ewma_config_negative = EwmaConfig {
        initialization_value: -0.1,
        ..EwmaConfig::default()
    };
    let result = EwmaPredictor::new(timer.clone(), config.clone(), ewma_config_negative);
    assert!(matches!(result, Err(EwmaError::InvalidInitValue(_))));

    // Test initialization value > 1 (invalid)
    let ewma_config_large = EwmaConfig {
        initialization_value: 1.5,
        ..EwmaConfig::default()
    };
    let result = EwmaPredictor::new(timer, config, ewma_config_large);
    assert!(matches!(result, Err(EwmaError::InvalidInitValue(_))));
}

#[test]
fn test_ewma_update_basic() {
    let timer = create_test_timer();
    let config = policy::constants::models::PHI_TINY_MOE.clone();
    let ewma_config = EwmaConfig {
        alpha: 0.5,
        initialization_value: 0.1,
    };

    let mut predictor = EwmaPredictor::new(timer, config, ewma_config).unwrap();

    // Simulate activation of expert 0 in layer 0
    let activated_experts = vec![0];
    let result = predictor.update_layer_activations(&activated_experts);
    assert!(result.is_ok());

    // Check that expert (0,0) has updated probability
    let expert_key = ExpertKey::new(0, 0);
    let probability = predictor.get_probability(expert_key);

    // Expected: (1-0.5)*0.1 + 0.5*1.0 = 0.05 + 0.5 = 0.55
    let expected = (1.0 - 0.5) * 0.1 + 0.5 * 1.0;
    assert!((probability - expected).abs() < 1e-10);
}

#[test]
fn test_ewma_update_multiple_experts() {
    let timer = create_test_timer();
    let config = policy::constants::models::PHI_TINY_MOE.clone();
    let ewma_config = EwmaConfig {
        alpha: 0.3,
        initialization_value: 0.2,
    };

    let mut predictor = EwmaPredictor::new(timer, config, ewma_config).unwrap();

    // Simulate activation of experts 0 and 2 in layer 0
    let activated_experts = vec![0, 2];
    let result = predictor.update_layer_activations(&activated_experts);
    assert!(result.is_ok());

    // Check activated experts
    let expert_0 = ExpertKey::new(0, 0);
    let expert_2 = ExpertKey::new(2, 0);
    let prob_0 = predictor.get_probability(expert_0);
    let prob_2 = predictor.get_probability(expert_2);

    // Expected for activated: (1-0.3)*0.2 + 0.3*1.0 = 0.14 + 0.3 = 0.44
    let expected_activated = (1.0 - 0.3) * 0.2 + 0.3 * 1.0;
    assert!((prob_0 - expected_activated).abs() < 1e-10);
    assert!((prob_2 - expected_activated).abs() < 1e-10);

    // Check non-activated expert
    let expert_1 = ExpertKey::new(1, 0);
    let prob_1 = predictor.get_probability(expert_1);

    // Expected for non-activated: (1-0.3)*0.2 + 0.3*0.0 = 0.14
    let expected_non_activated = (1.0 - 0.3) * 0.2;
    assert!((prob_1 - expected_non_activated).abs() < 1e-10);
}

#[test]
fn test_layer_probabilities() {
    let timer = create_test_timer();
    let config = policy::constants::models::PHI_TINY_MOE.clone();
    let ewma_config = EwmaConfig::default();

    let mut predictor = EwmaPredictor::new(timer, config, ewma_config).unwrap();

    // Update with some activations
    let activated_experts = vec![1, 3];
    let _result = predictor.update_layer_activations(&activated_experts);

    // Get all probabilities for layer 0
    let layer_probs = predictor.get_layer_probabilities(0);

    // Should have probabilities for all 4 experts in the layer
    assert_eq!(layer_probs.len(), 4);
    assert!(layer_probs.contains_key(&0));
    assert!(layer_probs.contains_key(&1));
    assert!(layer_probs.contains_key(&2));
    assert!(layer_probs.contains_key(&3));

    // Activated experts should have higher probabilities
    assert!(layer_probs[&1] > layer_probs[&0]);
    assert!(layer_probs[&3] > layer_probs[&2]);
}

#[test]
fn test_statistical_properties() {
    let timer = create_test_timer();
    let config = policy::constants::models::PHI_TINY_MOE.clone();
    let ewma_config = EwmaConfig {
        alpha: 0.2,
        initialization_value: 0.1,
    };

    let predictor = EwmaPredictor::new(timer, config, ewma_config).unwrap();

    // Test effective window size calculation
    let expected_window = (2.0 - 0.2) / 0.2; // = 9.0
    assert!((predictor.effective_window_size() - expected_window).abs() < 1e-10);

    // Test variance bound calculation
    let true_prob = 0.3;
    let expected_variance = true_prob * (1.0 - true_prob) * 0.2 / (2.0 - 0.2);
    assert!((predictor.variance_bound(true_prob) - expected_variance).abs() < 1e-10);
}

#[test]
fn test_model_specific_constructors() {
    let timer = create_test_timer();

    // Test GPT-OSS-20B constructor
    let predictor_20b = EwmaPredictor::for_gptoss20b(timer.clone());
    assert!(predictor_20b.is_ok());
    let predictor_20b = predictor_20b.unwrap();
    assert_eq!(predictor_20b.config().total_layers, 24);

    // Test Phi-Tiny-MoE constructor
    let predictor_phi = EwmaPredictor::for_phi_tiny_moe(timer);
    assert!(predictor_phi.is_ok());
    let predictor_phi = predictor_phi.unwrap();
    assert_eq!(predictor_phi.config().total_layers, 8);
}

#[test]
fn test_reset_functionality() {
    let timer = create_test_timer();
    let config = policy::constants::models::PHI_TINY_MOE.clone();
    let ewma_config = EwmaConfig::default();

    let mut predictor = EwmaPredictor::new(timer, config, ewma_config).unwrap();

    // Add some EWMA values
    let activated_experts = vec![0, 1];
    let _result = predictor.update_layer_activations(&activated_experts);
    assert!(predictor.num_tracked_experts() > 0);

    // Reset and check
    predictor.reset();
    assert_eq!(predictor.num_tracked_experts(), 0);

    // Check that probabilities return to initialization value
    let expert_key = ExpertKey::new(0, 0);
    let probability = predictor.get_probability(expert_key);
    assert_eq!(probability, predictor.initialization_value());
}

#[test]
fn test_expert_key_functionality() {
    // Test ExpertKey creation
    let key1 = ExpertKey::new(1, 2);
    assert_eq!(key1.expert_id, 1);
    assert_eq!(key1.layer_id, 2);

    // Test ExpertKey validation
    let config = policy::constants::models::PHI_TINY_MOE.clone();

    // Valid key
    let valid_key = ExpertKey::with_validation(0, 0, &config);
    assert!(valid_key.is_ok());

    // Invalid layer
    let invalid_layer = ExpertKey::with_validation(0, 10, &config);
    assert!(invalid_layer.is_err());

    // Invalid expert
    let invalid_expert = ExpertKey::with_validation(10, 0, &config);
    assert!(invalid_expert.is_err());

    // Test layer experts generation
    let layer_experts = ExpertKey::layer_experts(0, 4);
    assert_eq!(layer_experts.len(), 4);
    for (i, key) in layer_experts.iter().enumerate() {
        assert_eq!(key.expert_id, i);
        assert_eq!(key.layer_id, 0);
    }

    // Test all experts generation
    let all_experts = ExpertKey::all_experts(&config);
    let expected_count = config.total_layers * config.experts_per_layer;
    assert_eq!(all_experts.len(), expected_count);
}

#[test]
fn test_ewma_config_methods() {
    // Test default config
    let default_config = EwmaConfig::default();
    assert_eq!(default_config.alpha, policy::constants::ALPHA);
    assert_eq!(default_config.initialization_value, 0.1);

    // Test custom constructors
    let alpha_config = EwmaConfig::with_alpha(0.5);
    assert_eq!(alpha_config.alpha, 0.5);
    assert_eq!(alpha_config.initialization_value, 0.1);

    let init_config = EwmaConfig::with_init_value(0.2);
    assert_eq!(init_config.alpha, policy::constants::ALPHA);
    assert_eq!(init_config.initialization_value, 0.2);

    let custom_config = EwmaConfig::new(0.4, 0.3);
    assert_eq!(custom_config.alpha, 0.4);
    assert_eq!(custom_config.initialization_value, 0.3);

    // Test validation
    let valid_config = EwmaConfig::new(0.5, 0.5);
    assert!(valid_config.validate().is_ok());

    let invalid_alpha_config = EwmaConfig::new(0.0, 0.5);
    assert!(invalid_alpha_config.validate().is_err());

    let invalid_init_config = EwmaConfig::new(0.5, -0.1);
    assert!(invalid_init_config.validate().is_err());
}
