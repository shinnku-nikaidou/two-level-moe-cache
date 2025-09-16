use policy::constants::models::*;

#[test]
fn test_model_configs() {
    // Test GPT-OSS-20B configuration
    assert_eq!(GPT_OSS_20B.name, "gpt-oss-20b");
    assert_eq!(GPT_OSS_20B.total_layers, 24);
    assert_eq!(GPT_OSS_20B.experts_per_layer, 8);
    assert_eq!(GPT_OSS_20B.top_k, 2);

    // Test model lookup by name
    let config = get_model_config("gpt-oss-20b").unwrap();
    assert_eq!(config.total_layers, 24);

    // Test invalid model name
    assert!(get_model_config("nonexistent").is_none());
}

#[test]
fn test_available_models() {
    let models = available_models();
    assert_eq!(models.len(), 3);

    // Check all models are included
    let model_names: Vec<&str> = models.iter().map(|m| m.name).collect();
    assert!(model_names.contains(&"gpt-oss-20b"));
    assert!(model_names.contains(&"gpt-oss-120b"));
    assert!(model_names.contains(&"phi-tiny-moe"));
}

#[test]
fn test_model_config_validation() {
    for model in available_models() {
        // Validate all models have positive values
        assert!(
            model.total_layers > 0,
            "Model {} has invalid total_layers",
            model.name
        );
        assert!(
            model.experts_per_layer > 0,
            "Model {} has invalid experts_per_layer",
            model.name
        );
        assert!(model.top_k > 0, "Model {} has invalid top_k", model.name);
        assert!(
            model.top_k <= model.experts_per_layer,
            "Model {} has top_k > experts_per_layer",
            model.name
        );
    }
}

#[test]
fn test_timer_integration() {
    use policy::timer::Timer;

    // Test Timer with GPT-OSS-20B configuration
    let config = get_model_config("gpt-oss-20b").unwrap();
    let mut timer = Timer::new(config.total_layers).unwrap();

    // Verify timer works correctly with 24 layers
    assert_eq!(timer.total_layers(), 24);
    assert_eq!(timer.current_layer().unwrap(), 0); // Layer 0 at t=0

    // Test a full cycle through all layers
    for expected_layer in 0..24 {
        assert_eq!(timer.current_layer().unwrap(), expected_layer);
        timer.tick();
    }

    // Should be back to layer 0 after 24 ticks
    assert_eq!(timer.current_layer().unwrap(), 0);
}