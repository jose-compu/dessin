#!/usr/bin/env python3
"""
Start Web UI for Dessin Models
===============================

Starts a web interface where you can select and chat with any trained model.

Usage (from repo root):
    ./bin/python e2e/scripts/start_web_ui.py [--port PORT]

This will:
1. Start a test node
2. Create and train 3 sample models (if no models exist)
3. Launch the web UI
4. Open your browser to interact with models

Press Ctrl+C to stop.
"""

import argparse
import os
import sys
import time
import webbrowser

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dessin.runtime.config import DessinConfig
from dessin.nanochat.nanochat_integration import TrainingDevice
from dessin.runtime.node import DessinNode


def main():
    parser = argparse.ArgumentParser(description='Start Dessin Web UI')
    parser.add_argument('--port', type=int, default=8080, help='Port for web server (default: 8080)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    parser.add_argument('--quick-train', action='store_true', help='Quick training (5 iters) for demo')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧠 Dessin Web UI")
    print("=" * 80)
    print()
    
    node = None
    
    try:
        # Create test node
        print("Starting Dessin node...")
        config = DessinConfig.default()
        config.network.port = 9001
        config.network.bootstrap = True
        config.consensus.training_block_time_minutes = 1.0
        
        node = DessinNode(config)
        node.start()
        print(f"✓ Node started")
        print()
        
        # Create user with balance
        user_address = "web_ui_user"
        if hasattr(node.consensus, 'economic_system'):
            node.consensus.economic_system.balances[user_address] = 1000.0
        
        # Check if models already exist
        existing_models = list(node.model_manager.models.values())
        
        if not existing_models:
            print("No models found. Creating sample models...")
            print()
            
            # Create sample models
            model_configs = [
                {"name": "chat-assistant", "dataset": "FineWeb", "desc": "General chat"},
                {"name": "code-helper", "dataset": "CodeData", "desc": "Code assistance"},
                {"name": "doc-writer", "dataset": "Documentation", "desc": "Document generation"}
            ]
            
            iterations = 5 if args.quick_train else 10
            
            for cfg in model_configs:
                print(f"Creating {cfg['name']}...")
                model_id = node.nanochat_integration.create_model(
                    name=cfg["name"],
                    owner_address=user_address,
                    depth=10,
                    device_batch_size=4,
                    dataset_name=cfg["dataset"],
                    storage_blocks=100,
                    storage_payment=1.0,
                    training_payment_per_iteration=0.1,
                    training_device=TrainingDevice.CPU,
                    vocab_size=1024,
                    context_length=128,
                    max_steps=iterations
                )
                
                if model_id:
                    # Train the model
                    print(f"  Training {cfg['name']} ({iterations} iterations)...")
                    node.nanochat_integration.start_training(
                        model_id,
                        miner_address=node.address,
                        block_index=0
                    )
                    
                    for i in range(iterations):
                        success, metrics = node.nanochat_integration.train_iteration(model_id)
                        if success and hasattr(node.consensus, 'economic_system'):
                            node.consensus.economic_system.process_training_iteration_payment(
                                model_owner=user_address,
                                miner_address=node.address,
                                model_id=model_id,
                                iterations=1,
                                block_index=0
                            )
                    
                    job_id = f"job_{model_id}"
                    if job_id in node.nanochat_integration.training_jobs:
                        job = node.nanochat_integration.training_jobs[job_id]
                        print(f"  ✓ {cfg['name']}: {job.current_iteration} iters, loss {job.best_loss:.3f}")
                print()
        else:
            print(f"Found {len(existing_models)} existing model(s)")
            for model in existing_models:
                print(f"  • {model.name} ({model.parameters:,} params)")
            print()
        
        # Start web interface
        print("=" * 80)
        print("Starting web interface...")
        print("=" * 80)
        print()
        
        web_url = node.start_web_interface(port=args.port)
        
        if not web_url:
            print("❌ Failed to start web interface")
            return 1
        
        print()
        print("=" * 80)
        print("🌐 WEB UI READY!")
        print("=" * 80)
        print()
        print(f"URL: http://localhost:{args.port}")
        print()
        print("Features:")
        print("  • Select any trained model from dropdown")
        print("  • Chat with models in real-time")
        print("  • View model stats (parameters, loss, iterations)")
        print()
        
        # Get model list
        models = list(node.model_manager.models.values())
        if models:
            print(f"Available models ({len(models)}):")
            for model in models:
                print(f"  • {model.name}")
                job_id = f"job_{model.model_id}"
                if job_id in node.nanochat_integration.training_jobs:
                    job = node.nanochat_integration.training_jobs[job_id]
                    print(f"      {job.current_iteration} iterations, loss {job.best_loss:.3f}")
            print()
        
        # Open browser
        if not args.no_browser:
            try:
                print("Opening browser...")
                webbrowser.open(f'http://localhost:{args.port}')
                print("✓ Browser opened")
            except:
                print("(Could not open browser automatically)")
        
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 80)
        print()
        
        # Keep server running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if node:
            print("Stopping web interface...")
            try:
                node.stop_web_interface()
            except:
                pass
            print("Stopping node...")
            try:
                node.stop()
            except:
                pass
            print("✓ Stopped")


if __name__ == "__main__":
    sys.exit(main())
