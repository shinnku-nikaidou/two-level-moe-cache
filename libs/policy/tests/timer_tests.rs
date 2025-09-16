use policy::timer::Timer;

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

#[test]
fn test_from_gptoss20b() {
    // Test the GPT-OSS-20B specific constructor
    let timer = Timer::from_gptoss20b();

    // Should be configured with 24 layers
    assert_eq!(timer.total_layers(), 24);
    assert_eq!(timer.current_time(), 0);
    assert_eq!(timer.current_layer().unwrap(), 0); // Layer 0 at t=0

    // Test that it cycles through all 24 layers correctly
    let mut timer = Timer::from_gptoss20b();

    // Test first few layers
    for expected_layer in 0..24 {
        assert_eq!(timer.current_layer().unwrap(), expected_layer);
        timer.tick();
    }

    // After 24 ticks, should be back to layer 0
    assert_eq!(timer.current_layer().unwrap(), 0);
    assert_eq!(timer.current_time(), 24);

    // Test that layer visit counts work correctly
    assert_eq!(timer.layer_visit_count(0).unwrap(), 2); // Visited at t=0 and t=24
    assert_eq!(timer.layer_visit_count(23).unwrap(), 1); // Visited once at t=23
}

#[test]
fn test_from_gptoss120b() {
    // Test the GPT-OSS-120B specific constructor
    let timer = Timer::from_gptoss120b();

    // Should be configured with 36 layers
    assert_eq!(timer.total_layers(), 36);
    assert_eq!(timer.current_time(), 0);
    assert_eq!(timer.current_layer().unwrap(), 0); // Layer 0 at t=0

    // Test that it cycles through all 36 layers correctly
    let mut timer = Timer::from_gptoss120b();

    // Test a full cycle
    for expected_layer in 0..36 {
        assert_eq!(timer.current_layer().unwrap(), expected_layer);
        timer.tick();
    }

    // After 36 ticks, should be back to layer 0
    assert_eq!(timer.current_layer().unwrap(), 0);
    assert_eq!(timer.current_time(), 36);
}

#[test]
fn test_from_phi_tiny_moe() {
    // Test the Phi-Tiny-MoE specific constructor
    let timer = Timer::from_phi_tiny_moe();

    // Should be configured with 8 layers
    assert_eq!(timer.total_layers(), 8);
    assert_eq!(timer.current_time(), 0);
    assert_eq!(timer.current_layer().unwrap(), 0); // Layer 0 at t=0

    // Test that it cycles through all 8 layers correctly
    let mut timer = Timer::from_phi_tiny_moe();

    // Test a full cycle
    for expected_layer in 0..8 {
        assert_eq!(timer.current_layer().unwrap(), expected_layer);
        timer.tick();
    }

    // After 8 ticks, should be back to layer 0
    assert_eq!(timer.current_layer().unwrap(), 0);
    assert_eq!(timer.current_time(), 8);

    // Test multiple cycles
    for _ in 0..16 {
        timer.tick();
    }
    assert_eq!(timer.current_layer().unwrap(), 0); // Should be layer 0 at t=24 (8+16)
    assert_eq!(timer.current_time(), 24);
    assert_eq!(timer.layer_visit_count(0).unwrap(), 4); // Visited at t=0, t=8, t=16, t=24
}

#[test]
fn test_all_model_constructors() {
    // Test that all model constructors create valid timers
    let gpt20b = Timer::from_gptoss20b();
    let gpt120b = Timer::from_gptoss120b();
    let phi_tiny = Timer::from_phi_tiny_moe();

    // All should start at time 0 and layer 0
    assert_eq!(gpt20b.current_time(), 0);
    assert_eq!(gpt20b.current_layer().unwrap(), 0);

    assert_eq!(gpt120b.current_time(), 0);
    assert_eq!(gpt120b.current_layer().unwrap(), 0);

    assert_eq!(phi_tiny.current_time(), 0);
    assert_eq!(phi_tiny.current_layer().unwrap(), 0);

    // Check expected layer counts
    assert_eq!(gpt20b.total_layers(), 24);
    assert_eq!(gpt120b.total_layers(), 36);
    assert_eq!(phi_tiny.total_layers(), 8);
}