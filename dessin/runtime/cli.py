"""
Command Line Interface for DeSSIN

Provides CLI commands for local model queries and management.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn


def run_async(coro):
    """Helper function to run async code in CLI context"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're already in an event loop, create a task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No event loop running, we can use asyncio.run
        return asyncio.run(coro)

from ..models.model_manager import ModelManager
from .config import ModelConfig
from ..nanochat.local_model_runner import LocalModelRunner, QueryRequest, list_available_local_models
from ..distribution.huggingface_distributor import HuggingFaceDistributor
from ..distribution.bittorrent_distributor import BitTorrentDistributor
from ..distribution.hybrid_distributor import HybridDistributor

console = Console()


@click.group()
@click.version_option(version="0.1.0")
@click.option("--config", "-c", help="Path to configuration file")
@click.pass_context
def cli(ctx, config):
    """DeSSIN - Decentralized AI Model Distribution System"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config


@cli.group()
def model():
    """Model management commands"""
    pass


@model.command()
@click.option("--cache-dir", help="Model cache directory")
def list(cache_dir):
    """List available local models"""
    try:
        config = ModelConfig(model_cache_dir=cache_dir or "~/.dessin/models")
        model_manager = ModelManager(config)
        
        models = list_available_local_models(model_manager)
        
        if not models:
            console.print("No models found locally.", style="yellow")
            return
        
        table = Table(title="Available Local Models")
        table.add_column("Model ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Size (GB)", style="blue")
        table.add_column("Format", style="magenta")
        table.add_column("Parameters", style="yellow")
        table.add_column("Available", style="red")
        
        for model in models:
            table.add_row(
                model["model_id"],
                model["name"],
                f"{model['size_gb']:.2f}",
                model["format"],
                f"{model['parameters']:,}" if model["parameters"] else "Unknown",
                "✓" if model["available"] else "✗"
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"Error listing models: {e}", style="red")


@model.command()
@click.argument("model_id")
@click.option("--cache-dir", help="Model cache directory")
def info(model_id, cache_dir):
    """Show detailed information about a model"""
    try:
        config = ModelConfig(model_cache_dir=cache_dir or "~/.dessin/models")
        model_manager = ModelManager(config)
        
        model_info = model_manager.get_model_info(model_id)
        if not model_info:
            console.print(f"Model {model_id} not found.", style="red")
            return
        
        info_panel = Panel(
            f"[bold]Model ID:[/bold] {model_id}\n"
            f"[bold]Name:[/bold] {getattr(model_info, 'name', 'Unknown')}\n"
            f"[bold]Size:[/bold] {getattr(model_info, 'size_gb', 0):.2f} GB\n"
            f"[bold]Format:[/bold] {getattr(model_info, 'format', 'Unknown')}\n"
            f"[bold]Quantization:[/bold] {getattr(model_info, 'quantization', 'Unknown')}\n"
            f"[bold]Parameters:[/bold] {getattr(model_info, 'parameters', 0):,}\n"
            f"[bold]Path:[/bold] {getattr(model_info, 'model_path', 'Unknown')}\n"
            f"[bold]Available:[/bold] {'✓' if Path(getattr(model_info, 'model_path', '')).exists() else '✗'}",
            title=f"Model Information: {model_id}",
            border_style="blue"
        )
        
        console.print(info_panel)
        
    except Exception as e:
        console.print(f"Error getting model info: {e}", style="red")


@cli.group()
def query():
    """Local model query commands"""
    pass


@query.command()
@click.argument("model_id")
@click.argument("prompt")
@click.option("--max-tokens", "-m", default=512, help="Maximum tokens to generate")
@click.option("--temperature", "-t", default=0.7, help="Sampling temperature")
@click.option("--top-p", default=0.9, help="Top-p sampling parameter")
@click.option("--system-prompt", "-s", help="System prompt")
@click.option("--cache-dir", help="Model cache directory")
@click.option("--output", "-o", help="Output file for response")
def run(model_id, prompt, max_tokens, temperature, top_p, system_prompt, cache_dir, output):
    """Run a local model query"""
    
    async def run_query():
        try:
            config = ModelConfig(model_cache_dir=cache_dir or "~/.dessin/models")
            model_manager = ModelManager(config)
            runner = LocalModelRunner(model_manager)
            
            # Check if model exists
            model_info = model_manager.get_model_info(model_id)
            if not model_info:
                console.print(f"Model {model_id} not found locally.", style="red")
                return
            
            # Create query request
            request = QueryRequest(
                model_id=model_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                system_prompt=system_prompt
            )
            
            # Show progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Loading model and running inference...", total=None)
                
                # Run query
                response = await runner.query_model(request)
                progress.update(task, completed=True)
            
            if response:
                # Display response
                console.print("\n[bold green]Response:[/bold green]")
                console.print(response.response)
                
                # Display metrics
                console.print(f"\n[bold blue]Metrics:[/bold blue]")
                console.print(f"Tokens: {response.prompt_tokens} → {response.response_tokens} (total: {response.total_tokens})")
                console.print(f"Inference time: {response.inference_time:.2f}s")
                console.print(f"Model type: {response.model_type.value}")
                
                # Save to file if requested
                if output:
                    with open(output, 'w') as f:
                        json.dump({
                            "model_id": response.model_id,
                            "prompt": prompt,
                            "response": response.response,
                            "metrics": {
                                "prompt_tokens": response.prompt_tokens,
                                "response_tokens": response.response_tokens,
                                "total_tokens": response.total_tokens,
                                "inference_time": response.inference_time
                            },
                            "timestamp": response.timestamp
                        }, f, indent=2)
                    console.print(f"\n[green]Response saved to {output}[/green]")
            else:
                console.print("Query failed.", style="red")
                
        except Exception as e:
            console.print(f"Error running query: {e}", style="red")
    
    run_async(run_query())


@query.command()
@click.option("--cache-dir", help="Model cache directory")
def interactive(cache_dir):
    """Start interactive query session"""
    
    async def interactive_session():
        try:
            config = ModelConfig(model_cache_dir=cache_dir or "~/.dessin/models")
            model_manager = ModelManager(config)
            runner = LocalModelRunner(model_manager)
            
            # List available models
            models = list_available_local_models(model_manager)
            if not models:
                console.print("No models available for interactive session.", style="red")
                return
            
            console.print("[bold green]Available models:[/bold green]")
            for i, model in enumerate(models):
                console.print(f"{i+1}. {model['model_id']} ({model['name']})")
            
            # Select model
            while True:
                try:
                    choice = int(console.input("\nSelect model number: ")) - 1
                    if 0 <= choice < len(models):
                        selected_model = models[choice]["model_id"]
                        break
                    else:
                        console.print("Invalid choice. Please try again.", style="red")
                except ValueError:
                    console.print("Please enter a valid number.", style="red")
            
            console.print(f"\n[bold blue]Interactive session with {selected_model}[/bold blue]")
            console.print("Type 'quit' to exit, 'help' for commands.\n")
            
            while True:
                try:
                    user_input = console.input("[bold cyan]You:[/bold cyan] ")
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    elif user_input.lower() == 'help':
                        console.print("\n[bold]Commands:[/bold]")
                        console.print("  quit/exit/q - Exit session")
                        console.print("  help - Show this help")
                        console.print("  metrics - Show model metrics")
                        console.print("  clear - Clear conversation")
                        console.print("  <text> - Send query to model\n")
                        continue
                    elif user_input.lower() == 'metrics':
                        metrics = runner.get_model_metrics(selected_model)
                        if metrics:
                            console.print(f"\n[bold]Model Metrics:[/bold]")
                            console.print(f"Total queries: {metrics.total_queries}")
                            console.print(f"Total tokens: {metrics.total_tokens}")
                            console.print(f"Average inference time: {metrics.average_inference_time:.2f}s")
                            console.print(f"Success rate: {metrics.success_rate:.2%}\n")
                        continue
                    elif user_input.lower() == 'clear':
                        runner.clear_history()
                        console.print("Conversation history cleared.\n")
                        continue
                    
                    # Run query
                    request = QueryRequest(
                        model_id=selected_model,
                        prompt=user_input,
                        max_tokens=512,
                        temperature=0.7
                    )
                    
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                        transient=True
                    ) as progress:
                        task = progress.add_task("Generating response...", total=None)
                        response = await runner.query_model(request)
                        progress.update(task, completed=True)
                    
                    if response:
                        console.print(f"[bold green]Model:[/bold green] {response.response}\n")
                    else:
                        console.print("Query failed.\n")
                        
                except KeyboardInterrupt:
                    console.print("\nExiting...")
                    break
                except Exception as e:
                    console.print(f"Error: {e}", style="red")
            
        except Exception as e:
            console.print(f"Error in interactive session: {e}", style="red")
    
    run_async(interactive_session())


@cli.group()
def download():
    """Model download commands"""
    pass


@download.command()
@click.argument("model_id")
@click.option("--cache-dir", help="Model cache directory")
@click.option("--method", "-m", type=click.Choice(['hf', 'bittorrent', 'hybrid']), default='hybrid', help="Download method")
def model(model_id, cache_dir, method):
    """Download a model"""
    
    async def download_model():
        try:
            config = ModelConfig(model_cache_dir=cache_dir or "~/.dessin/models")
            model_manager = ModelManager(config)
            
            if method == 'hf':
                distributor = HuggingFaceDistributor(model_manager, force_mock=False)
                success, local_path, download_time = await distributor.download_model(model_id)
            elif method == 'bittorrent':
                distributor = BitTorrentDistributor(model_manager)
                # This would need torrent info or magnet link
                console.print("BitTorrent download requires torrent info or magnet link.", style="yellow")
                return
            else:  # hybrid
                distributor = HybridDistributor(model_manager)
                result = await distributor.download_model(model_id)
                success = result["success"]
                local_path = result.get("local_path")
                download_time = result.get("download_time", 0)
            
            if success:
                console.print(f"✓ Model {model_id} downloaded successfully", style="green")
                console.print(f"  Path: {local_path}")
                console.print(f"  Time: {download_time:.2f}s")
            else:
                console.print(f"❌ Failed to download model {model_id}", style="red")
                
        except Exception as e:
            console.print(f"Error downloading model: {e}", style="red")
    
    run_async(download_model())


@cli.group()
def upload():
    """Model upload commands"""
    pass


@upload.command()
@click.argument("model_id")
@click.argument("hf_repo_name")
@click.option("--description", "-d", help="Model description")
@click.option("--tags", "-t", multiple=True, help="Model tags")
@click.option("--private", is_flag=True, help="Make repository private")
@click.option("--cache-dir", help="Model cache directory")
def hf(model_id, hf_repo_name, description, tags, private, cache_dir):
    """Upload model to HuggingFace Hub"""
    
    async def upload_to_hf():
        try:
            config = ModelConfig(model_cache_dir=cache_dir or "~/.dessin/models")
            model_manager = ModelManager(config)
            distributor = HuggingFaceDistributor(model_manager, force_mock=False)
            
            result = await distributor.upload_model_to_hf(
                model_id=model_id,
                hf_repo_name=hf_repo_name,
                description=description or "",
                tags=list(tags) if tags else None,
                private=private
            )
            
            if result.success:
                console.print(f"✓ Model {model_id} uploaded to {hf_repo_name}", style="green")
                console.print(f"  URL: {result.hf_url}")
                console.print(f"  Size: {result.file_size_mb:.1f} MB")
                console.print(f"  Time: {result.upload_time:.2f}s")
            else:
                console.print(f"❌ Failed to upload model: {result.error_message}", style="red")
                
        except Exception as e:
            console.print(f"Error uploading model: {e}", style="red")
    
    run_async(upload_to_hf())


@cli.command()
@click.option("--cache-dir", help="Model cache directory")
def status(cache_dir):
    """Show system status"""
    try:
        config = ModelConfig(model_cache_dir=cache_dir or "~/.dessin/models")
        model_manager = ModelManager(config)
        runner = LocalModelRunner(model_manager)
        
        # Get metrics
        all_metrics = runner.get_all_metrics()
        memory_usage = runner.get_memory_usage()
        
        # Display status
        console.print("[bold blue]DeSSIN System Status[/bold blue]\n")
        
        # Models
        models = list_available_local_models(model_manager)
        console.print(f"[bold]Local Models:[/bold] {len(models)}")
        console.print(f"[bold]Loaded Models:[/bold] {len(runner.loaded_models)}")
        
        # Memory
        if memory_usage["gpu_memory"]:
            gpu = memory_usage["gpu_memory"]
            console.print(f"[bold]GPU Memory:[/bold] {gpu['allocated']:.2f} GB allocated, {gpu['cached']:.2f} GB cached")
        
        # Metrics
        if all_metrics:
            console.print(f"\n[bold]Model Metrics:[/bold]")
            for model_id, metrics in all_metrics.items():
                console.print(f"  {model_id}: {metrics.total_queries} queries, {metrics.success_rate:.1%} success rate")
        
        # Query history
        history = runner.get_query_history(10)
        if history:
            console.print(f"\n[bold]Recent Queries:[/bold] {len(history)}")
        
    except Exception as e:
        console.print(f"Error getting status: {e}", style="red")


def main():
    """Main CLI entry point"""
    cli()


if __name__ == "__main__":
    main()
