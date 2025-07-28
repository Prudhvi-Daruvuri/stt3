# Multi-Backend Usage Guide

## Backend Selection

### faster_whisper Backend (Default)
```bash
# Use default settings
python stt_server.py --backend faster_whisper

# Specify Whisper model size
python stt_server.py --backend faster_whisper --model base
python stt_server.py --backend faster_whisper --model large-v2

# With additional parameters
python stt_server.py --backend faster_whisper --model base --language en --input-device 1
```

**Available Models for faster_whisper:**
- `tiny`, `tiny.en`
- `base`, `base.en` 
- `small`, `small.en`
- `medium`, `medium.en`
- `large-v1`, `large-v2`, `large-v3`
- Custom Hugging Face models (e.g., `deepdml/faster-whisper-large-v3-turbo-ct2`)

### parakeet Backend
```bash
# Use default Parakeet model (nvidia/parakeet-tdt-0.6b-v2)
python stt_server.py --backend parakeet

# Use specific Parakeet model
python stt_server.py --backend parakeet --model nvidia/parakeet-tdt-0.6b-v2

# With additional parameters
python stt_server.py --backend parakeet --language en --input-device 1
```

**Requirements for parakeet:**
```bash
pip install nemo_toolkit[asr]
```

**Available Models for parakeet:**
- `nvidia/parakeet-tdt-0.6b-v2` (default)
- Other NeMo ASR models from NVIDIA NGC

## Model Parameter Behavior

### When using faster_whisper backend:
- The `--model` parameter directly specifies the Whisper model
- Standard Whisper model names are supported
- Custom Hugging Face models can be used

### When using parakeet backend:
- Standard Whisper model names (tiny, base, small, etc.) are automatically mapped to the default Parakeet model
- To use a specific Parakeet model, provide the full model name (e.g., `nvidia/parakeet-tdt-0.6b-v2`)
- Custom NeMo models can be specified by their full name

## Examples

### Basic Usage
```bash
# Default faster_whisper with tiny model
python stt_server.py

# Parakeet with default model
python stt_server.py --backend parakeet
```

### Advanced Usage
```bash
# High-accuracy faster_whisper
python stt_server.py --backend faster_whisper --model large-v2 --language en

# Parakeet with custom settings
python stt_server.py --backend parakeet --language en --beam_size 5 --input-device 1

# Custom model paths
python stt_server.py --backend faster_whisper --model deepdml/faster-whisper-large-v3-turbo-ct2
python stt_server.py --backend parakeet --model nvidia/parakeet-tdt-0.6b-v2
```

## Troubleshooting

### Parakeet Backend Issues
If you encounter model loading errors with Parakeet:

1. **Check available models:**
   ```python
   import nemo.collections.asr as nemo_asr
   print(nemo_asr.models.ASRModel.list_available_models())
   ```

2. **Install dependencies:**
   ```bash
   pip install nemo_toolkit[asr]
   ```

3. **Use default model:**
   ```bash
   python stt_server.py --backend parakeet  # Uses nvidia/parakeet-tdt-0.6b-v2
   ```

### faster_whisper Backend Issues
If you encounter issues with faster_whisper:

1. **Check CUDA availability:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Use CPU if needed:**
   ```bash
   python stt_server.py --backend faster_whisper --device cpu
   ```

## Performance Considerations

### faster_whisper
- **Pros**: Mature, stable, wide model selection
- **Cons**: May be slower for some use cases
- **Best for**: General-purpose transcription, established workflows

### parakeet  
- **Pros**: Advanced NVIDIA model, potentially better accuracy
- **Cons**: Requires NVIDIA GPU, fewer model options
- **Best for**: High-accuracy requirements, NVIDIA GPU environments
