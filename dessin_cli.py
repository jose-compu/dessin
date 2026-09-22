#!/usr/bin/env python3
"""
DeSSIN CLI - Command line interface for the DeSSIN blockchain node.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "chaincraft"))

from dessin import DessinNode
from dessin.runtime.config import DessinConfig


def cmd_start(args):
    """Start a DeSSIN node"""
    config = DessinConfig.from_env()
    
    if args.port:
        config.network.port = args.port
    if args.mining:
        config.consensus.block_time_minutes = 0.6  # ~36s cadence (was 0.01h) for testing
    
    node = DessinNode(config)
    
    try:
        success = node.start()
        if not success:
            print("Failed to start node")
            return 1
        
        print("DeSSIN node started successfully")
        print(f"Address: {node.address}")
        print(f"Port: {node.chaincraft_node.port}")
        print("Press Ctrl+C to stop...")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down node...")
        node.stop()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        node.stop()
        return 1


def cmd_info(args):
    """Get node information"""
    # This would connect to a running node
    print("Node info command - would connect to running node")
    return 0


def cmd_query(args):
    """Query a model"""
    print(f"Querying model {args.model_id}: {args.query}")
    # This would connect to a running node and submit query
    return 0


def cmd_upload(args):
    """Upload a model"""
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model file not found: {args.model_path}")
        return 1
    
    print(f"Uploading model {args.name} from {args.model_path}")
    # This would connect to a running node and upload model
    return 0


def cmd_models(args):
    """List available models"""
    print("Available models:")
    # This would connect to a running node and list models
    return 0


def cmd_balance(args):
    """Check balance"""
    address = args.address if hasattr(args, 'address') else "self"
    print(f"Balance for {address}: XXX DESSIN")
    # This would connect to a running node and get balance
    return 0


def cmd_test_network(args):
    """Start a test network with multiple nodes"""
    print("Starting test network with 3 nodes...")
    
    nodes = []
    base_port = 8000
    
    try:
        # Create 3 test nodes
        for i in range(3):
            config = DessinConfig.default()
            config.network.port = base_port + i
            config.consensus.block_time_minutes = 0.3  # ~18s cadence for testing
            
            node = DessinNode(config)
            success = node.start()
            
            if success:
                nodes.append(node)
                print(f"Node {i+1} started on port {base_port + i}")
                print(f"  Address: {node.address}")
                
                # Create a test model on first node
                if i == 0:
                    node._create_dummy_model()
                    models = node.list_models()
                    if models:
                        print(f"  Created test model: {models[0]['model_id']}")
            else:
                print(f"Failed to start node {i+1}")
        
        if not nodes:
            print("No nodes started successfully")
            return 1
        
        print(f"\nTest network running with {len(nodes)} nodes")
        print("Mining test blocks every few seconds...")
        print("Press Ctrl+C to stop...")
        
        # Let the network run
        start_time = time.time()
        while True:
            time.sleep(5)
            
            # Print periodic status
            elapsed = time.time() - start_time
            if elapsed > 30:  # Every 30 seconds
                for i, node in enumerate(nodes):
                    info = node.get_node_info()
                    chain_info = node.get_chain_info()
                    print(f"Node {i+1}: Block {chain_info['chain_length']}, "
                          f"Balance: {info['balance']:.3f}, "
                          f"Peers: {info['peers']}")
                start_time = time.time()
    
    except KeyboardInterrupt:
        print("\nShutting down test network...")
        for node in nodes:
            node.stop()
        return 0
    except Exception as e:
        print(f"Error in test network: {e}")
        for node in nodes:
            node.stop()
        return 1


def main():
    parser = argparse.ArgumentParser(description="DeSSIN Blockchain CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start a DeSSIN node")
    start_parser.add_argument("--port", type=int, help="Port to listen on")
    start_parser.add_argument("--mining", action="store_true", help="Enable mining")
    start_parser.set_defaults(func=cmd_start)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get node information")
    info_parser.set_defaults(func=cmd_info)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query a model")
    query_parser.add_argument("model_id", help="Model ID to query")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens")
    query_parser.set_defaults(func=cmd_query)
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload a model")
    upload_parser.add_argument("name", help="Model name")
    upload_parser.add_argument("model_path", help="Path to model file")
    upload_parser.add_argument("--storage-blocks", type=int, default=1000, help="Storage duration in blocks")
    upload_parser.set_defaults(func=cmd_upload)
    
    # Models command
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.set_defaults(func=cmd_models)
    
    # Balance command
    balance_parser = subparsers.add_parser("balance", help="Check balance")
    balance_parser.add_argument("--address", help="Address to check (default: self)")
    balance_parser.set_defaults(func=cmd_balance)
    
    # Test network command
    test_parser = subparsers.add_parser("test-network", help="Start a test network")
    test_parser.set_defaults(func=cmd_test_network)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
