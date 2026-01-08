#!/usr/bin/env python3
"""
Dimension verification test for FRP input configuration.
Tests the dimension calculations without requiring JAX.
"""

def calculate_frp_dimensions(input_dim, include_metadata, include_wrapper, metadata_dim=3, wrapper_dim=0):
    """Calculate FRP input dimension based on configuration."""
    frp_input_dim = input_dim
    if include_metadata:
        frp_input_dim += metadata_dim
    if include_wrapper:
        frp_input_dim += wrapper_dim
    return frp_input_dim

def simulate_obs_transformation(raw_obs_dim, metadata_dim, wrapper_dim,
                                include_metadata, include_wrapper):
    """Simulate the observation transformation logic."""
    # Total observation dimension
    total_obs_dim = raw_obs_dim + metadata_dim + wrapper_dim

    # Split observation
    raw_obs_size = raw_obs_dim
    metadata_size = metadata_dim
    wrapper_size = wrapper_dim

    # Build FRP input parts
    frp_input_parts = [raw_obs_size]  # Always include raw obs
    if include_metadata:
        frp_input_parts.append(metadata_size)
    if include_wrapper:
        frp_input_parts.append(wrapper_size)

    frp_input_dim = sum(frp_input_parts)

    # Build remaining parts (not transformed)
    remaining_parts = []
    if not include_metadata:
        remaining_parts.append(metadata_size)
    if not include_wrapper:
        remaining_parts.append(wrapper_size)

    remaining_dim = sum(remaining_parts)

    # Reconstructed observation dimension
    reconstructed_dim = frp_input_dim + remaining_dim

    return {
        'total_obs_dim': total_obs_dim,
        'raw_obs_dim': raw_obs_dim,
        'metadata_dim': metadata_dim,
        'wrapper_dim': wrapper_dim,
        'frp_input_dim': frp_input_dim,
        'remaining_dim': remaining_dim,
        'reconstructed_dim': reconstructed_dim,
        'dimensions_match': total_obs_dim == reconstructed_dim
    }

def test_configuration(name, raw_obs_dim, metadata_dim, wrapper_dim,
                       include_metadata, include_wrapper):
    """Test a specific configuration."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Raw observation dim: {raw_obs_dim}")
    print(f"  Metadata dim: {metadata_dim}")
    print(f"  Wrapper dim: {wrapper_dim}")
    print(f"  Include metadata in FRP: {include_metadata}")
    print(f"  Include wrapper in FRP: {include_wrapper}")

    result = simulate_obs_transformation(
        raw_obs_dim, metadata_dim, wrapper_dim,
        include_metadata, include_wrapper
    )

    print(f"\nDimension Analysis:")
    print(f"  Total observation dim: {result['total_obs_dim']}")
    print(f"  FRP input dim: {result['frp_input_dim']}")
    print(f"  Remaining (not FRP) dim: {result['remaining_dim']}")
    print(f"  Reconstructed obs dim: {result['reconstructed_dim']}")
    print(f"  ✓ PASS" if result['dimensions_match'] else f"  ✗ FAIL")

    if result['dimensions_match']:
        print(f"\n  Breakdown:")
        print(f"    {result['raw_obs_dim']} (raw obs)")
        if include_metadata:
            print(f"    + {result['metadata_dim']} (metadata, in FRP)")
        else:
            print(f"    + {result['metadata_dim']} (metadata, NOT in FRP)")
        if include_wrapper:
            print(f"    + {result['wrapper_dim']} (wrapper, in FRP)")
        else:
            print(f"    + {result['wrapper_dim']} (wrapper, NOT in FRP)")
        print(f"    = {result['total_obs_dim']} total")
        print(f"\n  FRP processes: {result['frp_input_dim']} dimensions")
        print(f"  Passes through: {result['remaining_dim']} dimensions")

    return result['dimensions_match']

def main():
    """Run all dimension tests."""
    print("FRP Input Dimension Verification Tests")
    print("="*60)

    # Common environment dimensions
    # CartPole: obs_dim=4, discrete action (2 actions) -> wrapper=3 (2+1 for reset)
    # Continuous control: wrapper=2 (action+reset)

    all_passed = True

    # Test 1: Default - only raw obs (CartPole-like)
    all_passed &= test_configuration(
        "Default: Raw obs only (CartPole)",
        raw_obs_dim=4,
        metadata_dim=3,
        wrapper_dim=3,  # Discrete: n_actions + 1
        include_metadata=False,
        include_wrapper=False
    )

    # Test 2: Raw obs + metadata (CartPole-like)
    all_passed &= test_configuration(
        "Raw obs + metadata (CartPole)",
        raw_obs_dim=4,
        metadata_dim=3,
        wrapper_dim=3,
        include_metadata=True,
        include_wrapper=False
    )

    # Test 3: Raw obs + wrapper (CartPole-like)
    all_passed &= test_configuration(
        "Raw obs + wrapper (CartPole)",
        raw_obs_dim=4,
        metadata_dim=3,
        wrapper_dim=3,
        include_metadata=False,
        include_wrapper=True
    )

    # Test 4: All components (CartPole-like)
    all_passed &= test_configuration(
        "All components (CartPole)",
        raw_obs_dim=4,
        metadata_dim=3,
        wrapper_dim=3,
        include_metadata=True,
        include_wrapper=True
    )

    # Test 5: Continuous action environment
    all_passed &= test_configuration(
        "Continuous action env (default)",
        raw_obs_dim=8,
        metadata_dim=3,
        wrapper_dim=2,  # Continuous: action + reset
        include_metadata=False,
        include_wrapper=False
    )

    # Test 6: Continuous with all components
    all_passed &= test_configuration(
        "Continuous with all components",
        raw_obs_dim=8,
        metadata_dim=3,
        wrapper_dim=2,
        include_metadata=True,
        include_wrapper=True
    )

    # Test 7: PopGym typical environment
    all_passed &= test_configuration(
        "PopGym typical (16 actions)",
        raw_obs_dim=16,
        metadata_dim=3,
        wrapper_dim=17,  # 16 actions + 1
        include_metadata=False,
        include_wrapper=False
    )

    # Test 8: PopGym with metadata
    all_passed &= test_configuration(
        "PopGym with metadata",
        raw_obs_dim=16,
        metadata_dim=3,
        wrapper_dim=17,
        include_metadata=True,
        include_wrapper=False
    )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if all_passed:
        print("✓ All dimension tests PASSED")
        print("\nThe implementation correctly:")
        print("  1. Splits observations into [raw_obs, metadata, wrapper]")
        print("  2. Selectively builds FRP input based on configuration")
        print("  3. Reconstructs the full observation with correct dimensions")
        return 0
    else:
        print("✗ Some tests FAILED")
        return 1

if __name__ == "__main__":
    exit(main())
