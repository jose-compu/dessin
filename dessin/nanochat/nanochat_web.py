"""
Nanochat Web Interface for DeSSIN
=================================

Serves nanochat models with a web interface similar to ChatGPT.
Reuses nanochat's ui.html for the frontend.
"""

import os
import json
import asyncio
from typing import Dict, Optional, Any
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import threading


class NanochatWebHandler(BaseHTTPRequestHandler):
    """HTTP handler for nanochat web interface"""
    
    # Class variables to share state
    model_manager = None
    nanochat_integration = None
    node = None
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/' or self.path == '/index.html':
            self._serve_chat_interface()
        elif self.path == '/api/models':
            self._serve_model_list()
        elif self.path.startswith('/api/model/'):
            model_id = self.path.split('/')[-1]
            self._serve_model_info(model_id)
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/api/chat':
            self._handle_chat()
        elif self.path == '/api/create_model':
            self._handle_create_model()
        else:
            self.send_error(404, "Not Found")
    
    def _serve_chat_interface(self):
        """Serve the chat interface HTML"""
        html = self._generate_chat_html()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def _generate_chat_html(self) -> str:
        """Generate chat interface HTML inspired by nanochat's ui.html"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeSSIN Nanochat</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        header {
            background: rgba(255, 255, 255, 0.95);
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #667eea;
            font-size: 1.5rem;
            font-weight: 600;
        }
        
        .subtitle {
            color: #666;
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }
        
        .container {
            flex: 1;
            max-width: 900px;
            width: 100%;
            margin: 2rem auto;
            padding: 0 1rem;
            display: flex;
            flex-direction: column;
        }
        
        .model-selector {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .model-selector select {
            width: 100%;
            padding: 0.75rem;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-size: 1rem;
            cursor: pointer;
        }
        
        .chat-container {
            flex: 1;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            max-height: 600px;
        }
        
        .messages {
            flex: 1;
            padding: 1.5rem;
            overflow-y: auto;
        }
        
        .message {
            margin-bottom: 1rem;
            padding: 1rem;
            border-radius: 10px;
            max-width: 80%;
        }
        
        .message.user {
            background: #667eea;
            color: white;
            margin-left: auto;
        }
        
        .message.assistant {
            background: #f0f0f0;
            color: #333;
        }
        
        .message.system {
            background: #fff3cd;
            color: #856404;
            max-width: 100%;
            text-align: center;
            font-size: 0.875rem;
        }
        
        .input-container {
            padding: 1rem;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 0.5rem;
        }
        
        .input-container input {
            flex: 1;
            padding: 0.75rem;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-size: 1rem;
        }
        
        .input-container button {
            padding: 0.75rem 1.5rem;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .input-container button:hover {
            background: #5568d3;
        }
        
        .input-container button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .model-info {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-size: 0.875rem;
            color: #666;
        }
        
        .loading {
            display: inline-block;
            width: 1rem;
            height: 1rem;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <header>
        <h1>🧠 DeSSIN Nanochat</h1>
        <p class="subtitle">Decentralized AI Chat powered by Nanochat models</p>
    </header>
    
    <div class="container">
        <div class="model-selector">
            <select id="modelSelect">
                <option value="">Loading models...</option>
            </select>
        </div>
        
        <div class="chat-container">
            <div class="messages" id="messages">
                <div class="message system">
                    Select a model above to start chatting
                </div>
            </div>
            
            <div class="input-container">
                <input 
                    type="text" 
                    id="messageInput" 
                    placeholder="Type your message..."
                    disabled
                />
                <button id="sendButton" disabled>Send</button>
            </div>
        </div>
        
        <div class="model-info" id="modelInfo">
            No model selected
        </div>
    </div>
    
    <script>
        let currentModel = null;
        let isProcessing = false;
        
        // Load models on page load
        async function loadModels() {
            try {
                const response = await fetch('/api/models');
                const data = await response.json();
                
                const select = document.getElementById('modelSelect');
                
                if (!data.success) {
                    // Error from server
                    select.innerHTML = '<option value="">Error loading models</option>';
                    addSystemMessage(`Error: ${data.error || 'Failed to load models'}`);
                    select.disabled = true;
                    return;
                }
                
                if (data.count === 0) {
                    // No models available yet
                    select.innerHTML = '<option value="">No models available yet</option>';
                    addSystemMessage('No models have been created yet. Train a model first using the 4-node test or create one through the API.');
                    select.disabled = true;
                    return;
                }
                
                // Models available - populate dropdown
                select.innerHTML = '<option value="">Select a model...</option>';
                
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.model_id;
                    option.textContent = `${model.name} (d${model.depth || '?'}, ${(model.parameters / 1e6).toFixed(1)}M params)`;
                    select.appendChild(option);
                });
                
                select.disabled = false;
                addSystemMessage(`Loaded ${data.count} model${data.count > 1 ? 's' : ''}`);
            } catch (error) {
                console.error('Error loading models:', error);
                const select = document.getElementById('modelSelect');
                select.innerHTML = '<option value="">Network error</option>';
                addSystemMessage('Network error: Unable to connect to server');
                select.disabled = true;
            }
        }
        
        // Handle model selection
        document.getElementById('modelSelect').addEventListener('change', async (e) => {
            const modelId = e.target.value;
            if (!modelId) {
                currentModel = null;
                document.getElementById('messageInput').disabled = true;
                document.getElementById('sendButton').disabled = true;
                document.getElementById('modelInfo').textContent = 'No model selected';
                return;
            }
            
            try {
                const response = await fetch(`/api/model/${modelId}`);
                const model = await response.json();
                currentModel = model;
                
                document.getElementById('messageInput').disabled = false;
                document.getElementById('sendButton').disabled = false;
                
                // Update model info
                const info = `Model: ${model.name} | Depth: d${model.depth || '?'} | Parameters: ${(model.parameters / 1e6).toFixed(1)}M | Format: ${model.format}`;
                document.getElementById('modelInfo').textContent = info;
                
                addSystemMessage(`Loaded model: ${model.name}`);
                
            } catch (error) {
                console.error('Error loading model info:', error);
                addSystemMessage('Error loading model information');
            }
        });
        
        // Handle sending messages
        async function sendMessage() {
            if (isProcessing || !currentModel) return;
            
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message
            addUserMessage(message);
            input.value = '';
            
            // Disable input while processing
            isProcessing = true;
            document.getElementById('sendButton').disabled = true;
            document.getElementById('messageInput').disabled = true;
            
            // Add loading indicator
            const loadingDiv = addAssistantMessage('Thinking... <span class="loading"></span>');
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        model_id: currentModel.model_id,
                        message: message,
                        max_tokens: 200
                    })
                });
                
                const data = await response.json();
                
                // Remove loading indicator
                loadingDiv.remove();
                
                // Add assistant response
                if (data.response) {
                    addAssistantMessage(data.response);
                } else if (data.error) {
                    addSystemMessage(`Error: ${data.error}`);
                }
                
            } catch (error) {
                console.error('Error sending message:', error);
                loadingDiv.remove();
                addSystemMessage('Error communicating with server');
            } finally {
                isProcessing = false;
                document.getElementById('sendButton').disabled = false;
                document.getElementById('messageInput').disabled = false;
                document.getElementById('messageInput').focus();
            }
        }
        
        document.getElementById('sendButton').addEventListener('click', sendMessage);
        document.getElementById('messageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        function addUserMessage(text) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message user';
            messageDiv.textContent = text;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function addAssistantMessage(html) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            messageDiv.innerHTML = html;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return messageDiv;
        }
        
        function addSystemMessage(text) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message system';
            messageDiv.textContent = text;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        // Load models on page load
        loadModels();
    </script>
</body>
</html>'''
    
    def _serve_model_list(self):
        """Serve list of available models"""
        try:
            if self.nanochat_integration:
                models = self.nanochat_integration.list_models()
            else:
                models = []
            
            # Return success response with models (even if empty)
            response = {
                'success': True,
                'models': models,
                'count': len(models)
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            # Return error response as JSON
            error_response = {
                'success': False,
                'models': [],
                'count': 0,
                'error': str(e)
            }
            self.send_response(200)  # Still 200 to allow JS to parse JSON
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())
    
    def _serve_model_info(self, model_id: str):
        """Serve information about a specific model"""
        try:
            if self.nanochat_integration:
                info = self.nanochat_integration.get_model_info(model_id)
                if info:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(info).encode())
                else:
                    self.send_error(404, "Model not found")
            else:
                self.send_error(500, "Nanochat integration not available")
                
        except Exception as e:
            self.send_error(500, str(e))
    
    def _handle_chat(self):
        """Handle chat message"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            model_id = data.get('model_id')
            message = data.get('message')
            max_tokens = data.get('max_tokens', 100)
            
            if not model_id or not message:
                self.send_error(400, "Missing model_id or message")
                return
            
            # Query the model - use nanochat integration for actual inference
            if self.nanochat_integration and model_id in self.nanochat_integration.model_configs:
                # Use simple GPT inference for nanochat models
                from .simple_gpt_inference import generate_from_nanochat_model
                
                config = self.nanochat_integration.model_configs[model_id]
                
                try:
                    response_text = generate_from_nanochat_model(
                        model_id=model_id,
                        prompt=message,
                        max_tokens=max_tokens,
                        temperature=0.8,
                        vocab_size=config.vocab_size,
                        context_length=config.context_length
                    )
                    
                    response_data = {
                        'response': response_text,
                        'error': None,
                        'tokens_used': len(response_text.split()),
                        'compute_cost': len(response_text.split()) * 0.001
                    }
                except Exception as e:
                    response_data = {
                        'response': None,
                        'error': str(e),
                        'tokens_used': 0,
                        'compute_cost': 0.0
                    }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode())
                
            elif self.model_manager:
                result = self.model_manager.query_model(
                    model_id=model_id,
                    query=message,
                    max_tokens=max_tokens
                )
                
                response_data = {
                    'response': result.response if result.success else None,
                    'error': result.error_message if not result.success else None,
                    'tokens_used': result.tokens_used,
                    'compute_cost': result.compute_cost
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode())
            else:
                self.send_error(500, "Model manager not available")
                
        except Exception as e:
            self.send_error(500, str(e))
    
    def _handle_create_model(self):
        """Handle model creation request"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            if self.nanochat_integration and self.node:
                model_id = self.nanochat_integration.create_model(
                    name=data.get('name', 'Unnamed Model'),
                    depth=data.get('depth', 20),
                    owner_address=self.node.address,
                    device_batch_size=data.get('device_batch_size', 32),
                    dataset_name=data.get('dataset_name', 'fineweb'),
                    storage_blocks=data.get('storage_blocks', 1000),
                    storage_payment=data.get('storage_payment', 10.0),
                    training_payment_per_iteration=data.get('training_payment_per_iteration', 0.001)
                )
                
                if model_id:
                    response_data = {
                        'success': True,
                        'model_id': model_id
                    }
                else:
                    response_data = {
                        'success': False,
                        'error': 'Failed to create model'
                    }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode())
            else:
                self.send_error(500, "Nanochat integration or node not available")
                
        except Exception as e:
            self.send_error(500, str(e))
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


class NanochatWebServer:
    """Web server for nanochat interface"""
    
    def __init__(
        self,
        model_manager,
        nanochat_integration,
        node=None,
        port: int = 8000,
        host: str = "0.0.0.0"
    ):
        self.model_manager = model_manager
        self.nanochat_integration = nanochat_integration
        self.node = node
        self.port = port
        self.host = host
        self.server = None
        self.server_thread = None
        
        # Set class variables for handler
        NanochatWebHandler.model_manager = model_manager
        NanochatWebHandler.nanochat_integration = nanochat_integration
        NanochatWebHandler.node = node
    
    def start(self):
        """Start the web server"""
        try:
            self.server = HTTPServer((self.host, self.port), NanochatWebHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            print(f"✓ Nanochat web interface started")
            print(f"  URL: http://{self.host}:{self.port}")
            print(f"  Access from local network: http://<your-ip>:{self.port}")
            
            return True
            
        except Exception as e:
            print(f"Error starting web server: {e}")
            return False
    
    def stop(self):
        """Stop the web server"""
        if self.server:
            self.server.shutdown()
            if self.server_thread:
                self.server_thread.join(timeout=5.0)
            print("✓ Nanochat web interface stopped")
    
    def get_url(self) -> str:
        """Get the server URL"""
        return f"http://{self.host}:{self.port}"
