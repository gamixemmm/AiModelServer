# Offline AI Chat Interface

A Python application that allows you to chat with the Mistral 7B AI model locally, without requiring an internet connection (after initial model download).

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with at least 8GB VRAM (optimized for RTX 4070)
- 16GB System RAM
- About 14GB of free disk space for the model

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python main.py
   ```
   Note: The first time you run the application, it will download the model (requires internet connection). Subsequent runs will use the cached model offline.

2. Start chatting with the AI model
3. Type 'quit' or 'exit' to end the conversation

## Features

- Completely offline operation (after initial model download)
- Uses Mistral 7B Instruct model, one of the best open-source models available
- 4-bit quantization for optimal memory usage
- Optimized for RTX 4070 and similar GPUs
- Interactive chat interface with colored output
- Advanced generation parameters for better responses

## Technical Details

This application uses the Mistral 7B Instruct model, which is currently one of the best performing open-source language models. The model:
- Uses 4-bit quantization to reduce VRAM usage
- Automatically utilizes GPU with optimized settings
- Requires approximately 14GB of storage space
- Uses about 8GB of VRAM during operation
- Features advanced generation parameters (temperature, top_p, top_k, repetition penalty)

## Alternative Models

You can also try other models by changing the `model_name` in `main.py`:
1. "meta-llama/Llama-2-7b-chat-hf" - Meta's Llama 2 model
2. "codellama/CodeLlama-7b-instruct-hf" - Specialized for coding tasks
3. "TinyLlama/TinyLlama-1.1B-Chat-v1.0" - Smaller model for less powerful hardware

## Troubleshooting

If you encounter "Out of Memory" errors:
1. Close other applications to free up VRAM and RAM
2. Try reducing the model's memory footprint by:
   - Using 8-bit quantization instead of 4-bit
   - Switching to a smaller model like TinyLlama
3. Make sure no other GPU-intensive applications are running 