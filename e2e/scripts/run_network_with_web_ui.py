#!/usr/bin/env python3
"""
Run Network with Web UI
========================

Trains models and keeps the network running with web UI for chatting.
The blockchain continues mining while you interact with trained models.
"""

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
    print("=" * 80)
    print("DeSSIN Network with Web UI")
    print("=" * 80)
    print()
    print("This will:")
    print("  1. Start a blockchain node")
    print("  2. Train 3 chat models (quick training)")
    print("  3. Start web UI for chatting")
    print("  4. Keep network running and mining")
    print()
    
    node = None
    
    try:
        # Step 1: Start Node
        print("Step 1: Starting Node")
        print("-" * 80)
        config = DessinConfig.default()
        config.network.port = 9001
        config.network.bootstrap = True
        config.consensus.training_block_time_minutes = 1.0
        
        node = DessinNode(config)
        node.start()
        print(f"✓ Node started: {node.address[:20]}...")
        print()
        
        # Create user with balance
        user_address = "chat_model_owner"
        if hasattr(node.consensus, 'economic_system'):
            node.consensus.economic_system.balances[user_address] = 1000.0
            print(f"✓ User balance: 1000.0 DESSIN")
        print()
        
        # Step 2: Create and Train Models
        print("Step 2: Creating and Training Models")
        print("-" * 80)
        print()
        
        models = []
        model_configs = [
            {
                "name": "friendly-assistant",
                "dataset": "Conversational",
                "desc": "Friendly chat assistant"
            },
            {
                "name": "technical-helper",
                "dataset": "Technical",
                "desc": "Technical support bot"
            },
            {
                "name": "creative-writer",
                "dataset": "Creative",
                "desc": "Creative writing assistant"
            }
        ]
        
        for i, cfg in enumerate(model_configs, 1):
            print(f"Creating model {i}/3: {cfg['name']}")
            
            model_id = node.nanochat_integration.create_model(
                name=cfg["name"],
                owner_address=user_address,
                depth=10,  # ~70M parameters
                device_batch_size=4,
                dataset_name=cfg["dataset"],
                storage_blocks=100,
                storage_payment=1.0,
                training_payment_per_iteration=0.1,
                training_device=TrainingDevice.CPU,
                vocab_size=1024,
                context_length=128,
                max_steps=10  # Quick training
            )
            
            if not model_id:
                print(f"  ❌ Failed to create {cfg['name']}")
                continue
            
            # Start training
            node.nanochat_integration.start_training(
                model_id,
                miner_address=node.address,
                block_index=0
            )
            
            # Train for 10 iterations
            print(f"  Training {cfg['name']} (10 iterations)...")
            for j in range(10):
                success, metrics = node.nanochat_integration.train_iteration(model_id)
                
                if success and hasattr(node.consensus, 'economic_system'):
                    node.consensus.economic_system.process_training_iteration_payment(
                        model_owner=user_address,
                        miner_address=node.address,
                        model_id=model_id,
                        iterations=1,
                        block_index=0
                    )
                
                if (j + 1) % 5 == 0:
                    loss = metrics.get('loss_after', 0)
                    print(f"    Iteration {j+1}/10: Loss={loss:.4f}")
            
            # Get final stats
            job_id = f"job_{model_id}"
            if job_id in node.nanochat_integration.training_jobs:
                job = node.nanochat_integration.training_jobs[job_id]
                models.append({
                    'id': model_id,
                    'name': cfg['name'],
                    'desc': cfg['desc'],
                    'iterations': job.current_iteration,
                    'loss': job.best_loss
                })
                print(f"  ✓ {cfg['name']}: {job.current_iteration} iters, loss {job.best_loss:.3f}")
            
            print()
        
        # Step 3: Start Web UI
        print("=" * 80)
        print("Step 3: Starting Web UI")
        print("=" * 80)
        print()
        
        web_port = 8080
        web_url = node.start_web_interface(port=web_port)
        
        if not web_url:
            print("❌ Failed to start web interface")
            return 1
        
        print()
        print("=" * 80)
        print("🌐 WEB UI IS READY!")
        print("=" * 80)
        print()
        print(f"URL: http://localhost:{web_port}")
        print()
        print("Available Models:")
        for model in models:
            print(f"  • {model['name']}")
            print(f"      {model['desc']}")
            print(f"      {model['iterations']} iterations, loss {model['loss']:.3f}")
        print()
        print("=" * 80)
        print()
        print("What you can do:")
        print("  1. Open http://localhost:8080 in your browser")
        print("  2. Select a model from the dropdown")
        print("  3. Chat with the trained model")
        print("  4. Switch between models to compare")
        print()
        print("The blockchain network is:")
        print("  ✓ Running in the background")
        print("  ✓ Mining blocks")
        print("  ✓ Processing transactions")
        print()
        print("=" * 80)
        print()
        
        # Try to open browser
        try:
            print("Opening browser...")
            webbrowser.open(f'http://localhost:{web_port}')
            print("✓ Browser opened automatically")
        except:
            print("(Could not open browser automatically)")
        
        print()
        print("=" * 80)
        print("NETWORK IS RUNNING")
        print("=" * 80)
        print()
        print("Commands:")
        print("  'status'  - Show network status")
        print("  'models'  - List models")
        print("  'balance' - Show user balance")
        print("  'blocks'  - Show blockchain info")
        print("  'help'    - Show commands")
        print("  'quit'    - Stop network and exit")
        print()
        print("Type a command or press Ctrl+C to stop:")
        print()
        
        # Keep network running
        while True:
            try:
                command = input("> ").strip().lower()
                
                if command in ['quit', 'exit', 'q', 'stop']:
                    print()
                    print("Shutting down...")
                    break
                
                elif command == 'status':
                    print()
                    print("Network Status:")
                    print(f"  Node address: {node.address[:40]}...")
                    print(f"  Blockchain height: {node.consensus.get_chain_length()}")
                    print(f"  Web UI: http://localhost:{web_port}")
                    print(f"  Mining: Active")
                    print()
                
                elif command == 'models':
                    print()
                    print("Trained Models:")
                    for model in models:
                        print(f"  • {model['name']}")
                        print(f"      ID: {model['id'][:30]}...")
                        print(f"      Iterations: {model['iterations']}")
                        print(f"      Loss: {model['loss']:.3f}")
                    print()
                
                elif command == 'balance':
                    if hasattr(node.consensus, 'economic_system'):
                        balance = node.consensus.economic_system.get_balance(user_address)
                        miner_balance = node.get_balance()
                        print()
                        print(f"User balance: {balance:.4f} DESSIN")
                        print(f"Miner earnings: {miner_balance:.4f} DESSIN")
                        print()
                    else:
                        print()
                        print("Balance information not available")
                        print()
                
                elif command == 'blocks':
                    chain_length = node.consensus.get_chain_length()
                    print()
                    print(f"Blockchain Info:")
                    print(f"  Height: {chain_length} blocks")
                    if chain_length > 0:
                        latest_block = node.consensus.chain[-1] if node.consensus.chain else None
                        if latest_block:
                            print(f"  Latest block: {latest_block.block_hash[:16]}...")
                            if hasattr(latest_block, 'model_id') and latest_block.model_id:
                                print(f"  Model in block: {latest_block.model_id[:30]}...")
                    print()
                
                elif command == 'help':
                    print()
                    print("Available Commands:")
                    print("  status  - Show network status")
                    print("  models  - List trained models")
                    print("  balance - Show balances")
                    print("  blocks  - Show blockchain info")
                    print("  help    - Show this help")
                    print("  quit    - Stop network and exit")
                    print()
                
                elif command == '':
                    continue
                
                else:
                    print()
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")
                    print()
            
            except EOFError:
                print()
                print("Shutting down...")
                break
        
        return 0
    
    except KeyboardInterrupt:
        print()
        print()
        print("Interrupted by user")
        return 0
    
    except Exception as e:
        print()
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if node:
            print()
            print("Stopping network...")
            try:
                node.stop_web_interface()
                print("✓ Web UI stopped")
            except:
                pass
            
            try:
                node.stop()
                print("✓ Node stopped")
            except:
                pass
            
            print()
            print("=" * 80)
            print("Network stopped. Thank you!")
            print("=" * 80)


if __name__ == "__main__":
    sys.exit(main())
