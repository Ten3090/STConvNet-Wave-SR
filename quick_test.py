import torch

from models.spatiotemporal_partial_fusion_sr import create_spatiotemporal_partial_fusion_sr


def main():
    print("="*70)
    print("Quick Test: STConvNet-Wave-SR Model")
    print("="*70)
    
    # Create model
    print("\n[1] Creating model...")
    model = create_spatiotemporal_partial_fusion_sr(
        num_frames=5,
        fusion_stages=[1, 2, 3, 4],
        use_temporal=True,
    )
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Create dummy input
    print("\n[2] Testing forward pass...")
    x = torch.randn(2, 5, 4, 8, 8)
    print(f"    Input shape:  {tuple(x.shape)}")
    
    # Forward pass
    with torch.no_grad():
        y = model(x)
    
    print(f"    Output shape: {tuple(y.shape)}")
    
    # Verify output shape
    expected_shape = (2, 1, 40, 40)
    if tuple(y.shape) != expected_shape:
        raise RuntimeError(f"Unexpected output shape: {tuple(y.shape)} != {expected_shape}")
    
    print("\n" + "="*70)
    print("✓ Quick test passed!")
    print("="*70)
    print("\nModel is ready for training.")


if __name__ == "__main__":
    main()
