# HuggingFace Model Diff Tool

A lightweight command-line tool to compare two HuggingFace models and visualize their differences in architecture, configuration, and weight structure - without downloading the full models.

## Features

- **Lightweight**: Only depends on `httpx` - no PyTorch or Transformers required
- **Fast**: Fetches only metadata (~KB) instead of downloading model weights (GB)
- **Comprehensive**: Compares configs, architectures, and tensor structures
- **Visual Diff**: Color-coded output showing additions, removals, and modifications
- **Compatibility Check**: Determines if models are structurally compatible

## Requirements

- Python 3.8+
- httpx

## Installation

### 1. Set Up Environment

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install httpx
```

### 3. (Optional) Set HuggingFace Token

For gated models (Llama, etc.):

```bash
export HF_TOKEN=hf_your_token_here
```

## Usage

### Basic Usage

```bash
python hf_diff.py <model_a> <model_b>
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `model_a` | First model ID (baseline) |
| `model_b` | Second model ID (to compare) |
| `--verbose`, `-v` | Show full detailed diff |
| `--token`, `-t` | HuggingFace token for gated models |
| `--no-color` | Disable colored output |
| `--json` | Output results as JSON |

## Examples

### Example 1: Compare GPT-2 Variants

```bash
python hf_diff.py gpt2 gpt2-medium
```

**Output:**
```
======================================================================
HuggingFace Model Diff
======================================================================

Model A: gpt2
Model B: gpt2-medium

Files: 26 vs 24
Tensors: 160 vs 316

──────────────────────────────────────────────────────────────────────
Configuration Diff (config.json)
──────────────────────────────────────────────────────────────────────

Key Parameters:
  [=] model_type: gpt2
  [=] vocab_size: 50257

Added (2):
  [+] n_special: 0
  [+] predict_special_tokens: True

──────────────────────────────────────────────────────────────────────
Weight Structure Diff
──────────────────────────────────────────────────────────────────────

Added Tensors (156):
  [+] h.12.* (13 tensors)
  [+] h.13.* (13 tensors)
  ... (more layers)

Modified Tensors (148):
  [~] h.0.attn.c_attn.weight
      shape: (768, 2304) → (1024, 3072)
  ...

──────────────────────────────────────────────────────────────────────
Compatibility Summary
──────────────────────────────────────────────────────────────────────
  Architecture Match:     ✓ Yes
  Hidden Size Match:      ✓ Yes
  Layer Count Match:      ✓ Yes
  Vocabulary Match:       ✓ Yes
  Weight Shapes Match:    ✗ No

  → Same architecture family but different configuration
======================================================================
```

### Example 2: Compare Base vs Instruct Model

```bash
python hf_diff.py mistralai/Mistral-7B-v0.1 mistralai/Mistral-7B-Instruct-v0.2
```

**Output:**
```
======================================================================
HuggingFace Model Diff
======================================================================

Model A: mistralai/Mistral-7B-v0.1
Model B: mistralai/Mistral-7B-Instruct-v0.2

Files: 14 vs 16
Tensors: 291 vs 291

──────────────────────────────────────────────────────────────────────
Configuration Diff (config.json)
──────────────────────────────────────────────────────────────────────

Key Parameters:
  [=] model_type: mistral
  [=] hidden_size: 4096
  [=] num_hidden_layers: 32
  [=] num_attention_heads: 32
  [=] num_key_value_heads: 8
  [=] vocab_size: 32000
  [=] max_position_embeddings: 32768

Added (1):
  [+] attention_dropout: 0.0

──────────────────────────────────────────────────────────────────────
Weight Structure Diff
──────────────────────────────────────────────────────────────────────
  Weight structures are identical

──────────────────────────────────────────────────────────────────────
Compatibility Summary
──────────────────────────────────────────────────────────────────────
  Architecture Match:     ✓ Yes
  Hidden Size Match:      ✓ Yes
  Layer Count Match:      ✓ Yes
  Vocabulary Match:       ✓ Yes
  Weight Shapes Match:    ✓ Yes

  → Models are structurally compatible (likely same base or fine-tune)
======================================================================
```

### Example 3: Compare Different Model Families

```bash
python hf_diff.py microsoft/phi-2 Qwen/Qwen2-1.5B
```

This will show significant differences in architecture, layer structure, and configuration.

### Example 4: Verbose Output

```bash
python hf_diff.py gpt2 gpt2-large --verbose
```

Shows all tensor modifications and full config diffs.

### Example 5: JSON Output for Scripting

```bash
python hf_diff.py gpt2 gpt2-medium --json > diff_result.json
```

**Output:**
```json
{
  "model_a": {
    "id": "gpt2",
    "config": { ... },
    "tensor_count": 160,
    "file_count": 26
  },
  "model_b": {
    "id": "gpt2-medium",
    "config": { ... },
    "tensor_count": 316,
    "file_count": 24
  },
  "config_diff": {
    "added": [["n_special", 0], ...],
    "removed": [],
    "modified": [...]
  },
  "tensor_diff": {
    "added": ["h.12.attn.c_attn.bias", ...],
    "removed": [],
    "modified_count": 148
  }
}
```

### Example 6: Gated Models

```bash
export HF_TOKEN=hf_your_token
python hf_diff.py meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf
```

Or use the `--token` flag:

```bash
python hf_diff.py meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf --token hf_xxx
```

## Understanding the Output

### Diff Symbols

| Symbol | Color | Meaning |
|--------|-------|---------|
| `[+]` | Green | Added in Model B |
| `[-]` | Red | Removed from Model A |
| `[~]` | Yellow | Modified between A and B |
| `[=]` | Gray | Unchanged |

### Compatibility Summary

The tool checks several aspects:

| Check | Description |
|-------|-------------|
| **Architecture Match** | Same model class (e.g., LlamaForCausalLM) |
| **Hidden Size Match** | Same embedding dimension |
| **Layer Count Match** | Same number of transformer layers |
| **Vocabulary Match** | Same tokenizer vocabulary size |
| **Weight Shapes Match** | All tensors have identical shapes |

### Verdicts

- **Structurally compatible**: Models can likely share weights (fine-tune relationship)
- **Same architecture family**: Same base architecture but different sizes
- **Different architectures**: Fundamentally different models

## Use Cases

1. **Verify Fine-tunes**: Check if a fine-tuned model maintains the same structure as the base
2. **Compare Versions**: See what changed between model versions (v0.1 vs v0.2)
3. **Size Comparison**: Understand differences between model sizes (7B vs 13B vs 70B)
4. **Adapter Compatibility**: Verify if LoRA adapters will be compatible
5. **Migration Planning**: Compare models before migrating inference infrastructure

## Limitations

- Only works with models that have safetensors files
- Does not compare actual weight values (only structure)
- Some config keys may use different names across model families

## Troubleshooting

### "Could not fetch repository files"

- Check if the model ID is correct
- The model may be private or gated - set `HF_TOKEN`

### Colors not showing

- Use a terminal that supports ANSI colors
- Or use `--no-color` flag

### Timeout errors

- Try again - HuggingFace servers may be slow
- Check your internet connection

## License

MIT License
