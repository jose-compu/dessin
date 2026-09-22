#!/usr/bin/env python3
"""
Multi-Layer Fine-Tuning Pipeline Demo
======================================

Demonstrates:
1. Creating a pipeline with multiple fine-tuning layers
2. Automatic progression based on conditions
3. Manual progression via transactions
4. Different progression conditions per layer
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig
from dessin.fine_tuning.fine_tuning_pipeline import (
    FineTuningPipelineManager,
    ProgressionCondition
)
from dessin.fine_tuning import FineTuningManager
import time


def demo_multi_layer_pipeline():
    print("=" * 80)
    print("Multi-Layer Fine-Tuning Pipeline Demo")
    print("=" * 80)
    print()
    
    # Setup node
    print("Setting up node...")
    config = DessinConfig.default()
    config.network.port = 9100
    config.network.bootstrap = True
    config.consensus.training_block_time_minutes = 1.0
    
    node = DessinNode(config)
    node.start()
    time.sleep(1)
    
    print(f"✓ Node started: {node.address[:20]}...")
    print()
    
    try:
        # Create model owner
        owner = "model_owner_address_123"
        node.consensus.economic_system.balances[owner] = 500.0
        
        # Create base model
        print("Creating base model...")
        base_model_id = node.nanochat_integration.create_model(
            name="base-gpt-d10",
            owner_address=owner,
            depth=10,
            device_batch_size=4,
            dataset_name="PreTraining",
            storage_blocks=100,
            storage_payment=10.0,
            training_payment_per_iteration=0.1
        )
        
        print(f"✓ Base model: {base_model_id}")
        print()
        
        # Create pipeline manager
        print("Initializing pipeline manager...")
        fine_tuning_manager = FineTuningManager(node.model_manager)
        pipeline_manager = FineTuningPipelineManager(
            node.model_manager,
            fine_tuning_manager
        )
        print()
        
        # Register datasets for each layer
        print("Registering fine-tuning datasets...")
        datasets = {}
        
        # Create dummy dataset files
        import tempfile
        import os
        
        for i, name in enumerate(['general', 'domain_specific', 'task_specific']):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                # Write dummy data
                for j in range(100):
                    f.write(f'{{"text": "Example {j} for {name} fine-tuning"}}\n')
                dataset_path = f.name
            
            dataset_id = fine_tuning_manager.register_dataset(
                dataset_name=f"{name}_dataset",
                dataset_path=dataset_path,
                dataset_format="jsonl",
                owner=owner,
                description=f"Dataset for {name} fine-tuning"
            )
            
            datasets[name] = dataset_id
            os.unlink(dataset_path)  # Clean up
        
        print(f"✓ Registered {len(datasets)} datasets")
        print()
        
        # Define multi-layer pipeline
        print("Creating 3-layer fine-tuning pipeline...")
        layer_configs = [
            {
                'name': 'Layer 1: General Adaptation',
                'description': 'Broad domain adaptation',
                'dataset_id': datasets['general'],
                'learning_rate': 3e-4,
                'num_iterations': 50,
                'batch_size': 4,
                'progression_condition': 'iteration_count',
                'condition_value': 50  # Auto-progress after 50 iterations
            },
            {
                'name': 'Layer 2: Domain Specialization',
                'description': 'Specialized domain knowledge',
                'dataset_id': datasets['domain_specific'],
                'learning_rate': 1e-4,
                'num_iterations': 100,
                'batch_size': 4,
                'progression_condition': 'loss_threshold',
                'condition_value': 7.0  # Auto-progress when loss < 7.0
            },
            {
                'name': 'Layer 3: Task-Specific Fine-Tuning',
                'description': 'Final task optimization',
                'dataset_id': datasets['task_specific'],
                'learning_rate': 5e-5,
                'num_iterations': 200,
                'batch_size': 4,
                'progression_condition': 'manual_only',
                'condition_value': 0  # Manual progression only
            }
        ]
        
        pipeline_id = pipeline_manager.create_pipeline(
            name="Production Fine-Tuning Pipeline",
            base_model_id=base_model_id,
            owner=owner,
            layer_configs=layer_configs,
            max_cost_dessin=200.0,
            description="Multi-stage fine-tuning for production model"
        )
        
        print(f"✓ Pipeline created: {pipeline_id}")
        print()
        
        # Start pipeline
        print("=" * 80)
        print("Starting Pipeline Training")
        print("=" * 80)
        print()
        
        pipeline_manager.start_pipeline(pipeline_id)
        
        # Training loop
        print("Training Layer 1: General Adaptation (auto-progress at 50 iterations)...")
        print()
        
        iteration_count = 0
        max_iterations = 200
        
        while iteration_count < max_iterations:
            status = pipeline_manager.get_pipeline_status(pipeline_id)
            
            if status['status'] == 'completed':
                print()
                print("✓ Pipeline completed!")
                break
            
            # Train one iteration
            success, metrics = pipeline_manager.train_iteration(pipeline_id)
            
            if not success:
                print(f"Training failed: {metrics.get('error')}")
                break
            
            iteration_count += 1
            
            # Print progress
            print(f"Iter {iteration_count:3d} | Layer {metrics['layer_index']}: {metrics['layer_name']}")
            print(f"        Loss: {metrics['loss_after']:.4f} (↓ {metrics['loss_improvement']:.4f})")
            print(f"        Best: {metrics['best_loss']:.4f}")
            print(f"        Cost: {metrics['total_cost']:.2f} DESSIN")
            
            # Check progression
            if 'progressed_to_layer' in metrics:
                print()
                print(f"  ⚡ AUTO-PROGRESSED to Layer {metrics['progressed_to_layer']}: {metrics['next_layer_name']}")
                print()
                time.sleep(1)
            
            if metrics.get('pipeline_completed'):
                print()
                print("✓ All layers completed!")
                break
            
            # Simulate some processing time
            time.sleep(0.1)
            
            # Show cumulative progress every 25 iterations
            if iteration_count % 25 == 0:
                current_layer = status['current_layer']
                print(f"        Progress: {current_layer['iterations_completed']}/{current_layer['max_iterations']} iterations")
                
                if status['current_layer_index'] == 2:  # Layer 3 (manual only)
                    print(f"        ⚠️  Manual progression required for this layer")
                print()
            
            # Demonstrate manual progression for Layer 3
            if status['current_layer_index'] == 2:  # Layer 3
                current_layer_obj = pipeline_manager.pipelines[pipeline_id].get_current_layer()
                
                # Manual progress after 30 iterations of layer 3
                if current_layer_obj.iterations_completed >= 30 and status['status'] != 'completed':
                    print()
                    print("=" * 80)
                    print("Manual Progression Trigger")
                    print("=" * 80)
                    print()
                    print("Owner sends blockchain transaction to progress to next layer...")
                    print("(In production: LayerProgressionTransaction would be mined)")
                    print()
                    
                    pipeline_manager.manual_progress_layer(pipeline_id, force=True)
                    print()
        
        # Final status
        print()
        print("=" * 80)
        print("Final Pipeline Status")
        print("=" * 80)
        print()
        
        final_status = pipeline_manager.get_pipeline_status(pipeline_id)
        
        print(f"Pipeline: {final_status['name']}")
        print(f"Status: {final_status['status']}")
        print(f"Total Iterations: {final_status['total_iterations']}")
        print(f"Total Cost: {final_status['total_cost']:.2f} DESSIN")
        print()
        
        print("Layer Summary:")
        for layer_info in final_status['layers']:
            status_icon = "✓" if layer_info['status'] == 'completed' else "○"
            print(f"  {status_icon} Layer {layer_info['index']}: {layer_info['name']}")
            print(f"      Iterations: {layer_info['iterations_completed']}/{layer_info['max_iterations']}")
            print(f"      Best Loss: {layer_info['best_loss']:.4f}")
            print(f"      Condition: {layer_info['progression_condition']}")
            print(f"      Status: {layer_info['status']}")
            print()
        
        print("=" * 80)
        print("Pipeline Workflow Demonstrated:")
        print("=" * 80)
        print()
        print("✓ Layer 1: Auto-progressed after 50 iterations (iteration_count)")
        print("✓ Layer 2: Auto-progressed when loss < 7.0 (loss_threshold)")
        print("✓ Layer 3: Manually progressed via transaction (manual_only)")
        print()
        print("Model owner has full control over pipeline progression!")
        print()
        
    finally:
        print("Stopping node...")
        node.stop()
        print("✓ Done")


if __name__ == "__main__":
    demo_multi_layer_pipeline()
