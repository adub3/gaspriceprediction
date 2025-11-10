"""
CARD Model Evaluation Test Script
tests/cardeval.py - Enhanced version with all fixes

Usage:
    python -m tests.cardeval --example basic --csv data/merged_data.csv --date_col date
    python -m tests.cardeval --example full --csv data/merged_data.csv --epochs 50
    python -m tests.cardeval --example data --csv data/merged_data.csv
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from models.card.vibes import (
        load_and_prepare_data_for_card,
        train_card_model,
        evaluate_card_model,
        Model
    )
    print("✓ Successfully imported CARD model components")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nMake sure models/card/vibes.py exists and is valid.")
    sys.exit(1)


def test_basic(args):
    """Quick test with minimal training"""
    print("\n" + "="*60)
    print("BASIC TEST - Quick Sanity Check")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from: {args.csv}")
    try:
        train_loader, val_loader, test_loader, config, normalizer, train_data = \
            load_and_prepare_data_for_card(
                csv_path=args.csv,
                date_col=args.date_col,
                value_cols=args.value_cols,
                seq_len=96,
                pred_len=96,
                label_len=48,
                train_ratio=0.7,
                val_ratio=0.1,
                batch_size=args.batch_size,
                volatility_window=args.volatility_window,
            )
        print("✓ Data loaded successfully")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create model
    print("\nCreating model...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        model = Model(config, normalizer=normalizer)
        print(f"✓ Model created ({sum(p.numel() for p in model.parameters()):,} parameters)")
        
        # Print architecture details
        print(f"\nArchitecture details:")
        print(f"  Patch len: {config.patch_len}, Stride: {config.stride}")
        print(f"  EMA enabled: {config.use_ema}")
        print(f"  Hybrid head: {config.use_hybrid_head}")
        print(f"  Bypass over-channel: {config.enc_in <= 2}")
        print(f"  Derivative channels: {config.normalize_deltas}")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Quick training (3 epochs)
    print("\nRunning quick training (3 epochs)...")
    try:
        trained_model = train_card_model(
            model,
            train_loader,
            val_loader,
            config,
            epochs=3,
            lr=1e-3,
            device=device,
            save_path="checkpoints/card_basic_test.pt",
            patience=10,
            loss_type='uniform'
        )
        print("✓ Training completed")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Quick evaluation
    print("\nEvaluating on test set...")
    try:
        os.makedirs("results_test", exist_ok=True)
        metrics, y_true, y_pred = evaluate_card_model(
            trained_model,
            test_loader,
            config,
            outdir="results_test/",
            device=device,
            train_data=train_data
        )
        print("✓ Evaluation completed")
        
        print("\n" + "="*60)
        print("BASIC TEST RESULTS")
        print("="*60)
        print(f"MSE:  {metrics['MSE']:.6f}")
        print(f"MAE:  {metrics['MAE']:.6f}")
        print(f"RMSE: {metrics['RMSE']:.6f}")
        print(f"sMAPE: {metrics['sMAPE']:.2f}%")
        print(f"MASE: {metrics['MASE']:.4f}")
        print(f"Direction Accuracy: {metrics['DirectionAccuracy']:.2f}%")
        
        # Show per-horizon metrics
        if 'per_horizon' in metrics:
            print("\nPer-Horizon Metrics:")
            for horizon, h_metrics in metrics['per_horizon'].items():
                print(f"  {horizon}: DA={h_metrics['DirectionAccuracy']:.2f}% MAE={h_metrics['MAE']:.4f}")
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return


def test_full(args):
    """Full training run"""
    print("\n" + "="*60)
    print("FULL TEST - Complete Training")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from: {args.csv}")
    train_loader, val_loader, test_loader, config, normalizer, train_data = \
        load_and_prepare_data_for_card(
            csv_path=args.csv,
            date_col=args.date_col,
            value_cols=args.value_cols,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            label_len=args.label_len,
            train_ratio=0.7,
            val_ratio=0.1,
            batch_size=args.batch_size,
            volatility_window=args.volatility_window
        )
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    model = Model(config, normalizer=normalizer)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Full training
    print(f"\nTraining for {args.epochs} epochs...")
    os.makedirs("checkpoints", exist_ok=True)
    trained_model = train_card_model(
        model,
        train_loader,
        val_loader,
        config,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=f"checkpoints/card_model_{args.epochs}ep.pt",
        patience=args.patience,
        loss_type=args.loss_type
    )
    
    # Full evaluation
    print("\nFinal evaluation...")
    os.makedirs(args.outdir, exist_ok=True)
    metrics, y_true, y_pred = evaluate_card_model(
        trained_model,
        test_loader,
        config,
        outdir=args.outdir,
        device=device,
        train_data=train_data
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, dict):
            continue
        print(f"{key}: {value:.6f}")
    
    # Show per-horizon metrics
    if 'per_horizon' in metrics:
        print("\nPer-Horizon Metrics:")
        for horizon, h_metrics in metrics['per_horizon'].items():
            print(f"  {horizon}: DA={h_metrics['DirectionAccuracy']:.2f}% MAE={h_metrics['MAE']:.4f}")


def test_inference(args):
    """Test inference with saved model"""
    print("\n" + "="*60)
    print("INFERENCE TEST - Load Saved Model")
    print("="*60)
    
    if not os.path.exists(args.checkpoint):
        print(f"✗ Checkpoint not found: {args.checkpoint}")
        print("Run training first with --example basic or --example full")
        return
    
    # Load data
    print(f"\nLoading data from: {args.csv}")
    train_loader, val_loader, test_loader, config, normalizer, train_data = \
        load_and_prepare_data_for_card(
            csv_path=args.csv,
            date_col=args.date_col,
            value_cols=args.value_cols,
            seq_len=96,
            pred_len=96,
            label_len=48,
            batch_size=32,
            volatility_window=args.volatility_window
        )
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create and load model
    model = Model(checkpoint['config'], normalizer=normalizer)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"✓ Model loaded (trained for {checkpoint['epoch']+1} epochs)")
    print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
    
    # Evaluate
    os.makedirs(args.outdir, exist_ok=True)
    metrics, y_true, y_pred = evaluate_card_model(
        model,
        test_loader,
        config,
        outdir=args.outdir,
        device=device,
        train_data=train_data
    )
    
    print("\n✓ Inference completed")


def test_data_only(args):
    """Test data loading only"""
    print("\n" + "="*60)
    print("DATA TEST - Verify Data Loading")
    print("="*60)
    
    print(f"\nLoading data from: {args.csv}")
    try:
        train_loader, val_loader, test_loader, config, normalizer, train_data = \
            load_and_prepare_data_for_card(
                csv_path=args.csv,
                date_col=args.date_col,
                value_cols=args.value_cols,
                seq_len=96,
                pred_len=96,
                label_len=48,
                batch_size=8
            )
        
        print("\n✓ Data loading successful!")
        print(f"\nDataset sizes:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        print(f"\nSample batch:")
        batch = next(iter(train_loader))
        x_enc, x_mark_enc, x_dec, x_mark_dec, y = batch
        print(f"  x_enc: {x_enc.shape}")
        print(f"  x_mark_enc: {x_mark_enc.shape}")
        print(f"  x_dec: {x_dec.shape}")
        print(f"  x_mark_dec: {x_mark_dec.shape}")
        print(f"  y: {y.shape}")
        
        print(f"\nData ranges:")
        print(f"  x_enc: [{x_enc.min():.4f}, {x_enc.max():.4f}]")
        print(f"  y: [{y.min():.4f}, {y.max():.4f}]")
        
        print(f"\nNormalizer stats:")
        print(f"  Mean: {normalizer.mean.flatten()}")
        print(f"  Std: {normalizer.std.flatten()}")
        
        print(f"\nConfig details:")
        print(f"  Patch len: {config.patch_len}, Stride: {config.stride}")
        print(f"  EMA: {config.use_ema}, Hybrid head: {config.use_hybrid_head}")
        print(f"  Loss weights: Huber={config.loss_huber_weight}, Slope={config.loss_slope_weight}")
        print(f"  Teacher forcing: {config.teacher_forcing_start} → {config.teacher_forcing_end}")
        print(f"  Autoregressive: {config.autoregressive}, AR steps: {config.ar_steps}")
        
    except Exception as e:
        print(f"\n✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='CARD Model Evaluation - Enhanced')
    
    # Test mode
    parser.add_argument('--example', type=str, default='basic',
                       choices=['basic', 'full', 'inference', 'data'],
                       help='Test mode: basic (3 epochs), full (complete), inference (load model), data (load only)')
    
    # Data arguments
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV file')
    parser.add_argument('--date_col', type=str, default='date',
                       help='Date column name')
    parser.add_argument('--value_cols', type=str, nargs='+', default=None,
                       help='Value columns (default: auto-detect numeric)')
    
    # Model arguments
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--volatility_window', type=int, default=7,
                       help='Window size for volatility-based normalization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--loss_type', type=str, default='uniform',
                       choices=['uniform', 'linear', 'sqrt'])
    
    # Output arguments
    parser.add_argument('--outdir', type=str, default='results_card/')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/card_basic_test.pt')
    
    args = parser.parse_args()
    
    # Validate CSV exists
    if not os.path.exists(args.csv):
        print(f"✗ CSV file not found: {args.csv}")
        sys.exit(1)
    
    print("="*60)
    print("CARD MODEL EVALUATION SCRIPT - ENHANCED")
    print("="*60)
    print(f"Mode: {args.example}")
    print(f"CSV: {args.csv}")
    print(f"Date column: {args.date_col}")
    print("\nEnhancements:")
    print("  ✓ Overlapping causal patches (stride=4)")
    print("  ✓ Multi-scale conv front-end")
    print("  ✓ Hybrid local+global output head")
    print("  ✓ Derivative channels (Δx, Δ²x)")
    print("  ✓ Trend-aware loss (Huber + Slope)")
    print("  ✓ Scheduled teacher forcing")
    print("  ✓ Conditional EMA (disabled for prices)")
    print("  ✓ Over-channel bypass (enc_in <= 2)")
    print("  ✓ MASE & sMAPE metrics")
    print("  ✓ Per-horizon directional accuracy")
    
    # Run selected test
    if args.example == 'basic':
        test_basic(args)
    elif args.example == 'full':
        test_full(args)
    elif args.example == 'inference':
        test_inference(args)
    elif args.example == 'data':
        test_data_only(args)
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()