#!/usr/bin/env python3
"""
HuggingFace Model Diff Tool
Compare two HuggingFace models to see architecture and weight differences.
Only depends on httpx.
"""

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

# ANSI color codes
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = ""
        cls.CYAN = cls.BOLD = cls.DIM = cls.RESET = ""


HF_BASE_URL = "https://huggingface.co"
HF_API_URL = "https://huggingface.co/api/models"


def get_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Get HTTP headers with optional HF token."""
    headers = {"User-Agent": "hf-diff/1.0"}
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@dataclass
class ModelMetadata:
    """Container for model metadata."""
    model_id: str
    config: Dict[str, Any] = field(default_factory=dict)
    generation_config: Dict[str, Any] = field(default_factory=dict)
    tensor_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    error: Optional[str] = None


def fetch_json(client: httpx.Client, url: str) -> Optional[Dict]:
    """Fetch and parse JSON from URL."""
    try:
        resp = client.get(url)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def fetch_safetensors_header(client: httpx.Client, url: str) -> Optional[Dict]:
    """Fetch only the metadata header from a safetensors file."""
    try:
        resp = client.get(url, headers={"Range": "bytes=0-7"})
        if resp.status_code not in (200, 206):
            return None
        
        header_size = struct.unpack("<Q", resp.content)[0]
        if header_size > 100 * 1024 * 1024:
            return None
        
        resp = client.get(url, headers={"Range": f"bytes=8-{8 + header_size - 1}"})
        if resp.status_code not in (200, 206):
            return None
        
        return json.loads(resp.content.decode("utf-8"))
    except Exception:
        return None


def get_repo_files(client: httpx.Client, model_id: str) -> List[str]:
    """Get all files in the repository."""
    url = f"{HF_API_URL}/{model_id}"
    data = fetch_json(client, url)
    if data and "siblings" in data:
        return [f["rfilename"] for f in data["siblings"]]
    return []


def fetch_model_metadata(client: httpx.Client, model_id: str) -> ModelMetadata:
    """Fetch all relevant metadata for a model."""
    metadata = ModelMetadata(model_id=model_id)
    
    # Get file list
    metadata.files = get_repo_files(client, model_id)
    if not metadata.files:
        metadata.error = "Could not fetch repository files (404 or access denied)"
        return metadata
    
    # Fetch config.json
    config_url = f"{HF_BASE_URL}/{model_id}/resolve/main/config.json"
    metadata.config = fetch_json(client, config_url) or {}
    
    # Fetch generation_config.json if exists
    if "generation_config.json" in metadata.files:
        gen_url = f"{HF_BASE_URL}/{model_id}/resolve/main/generation_config.json"
        metadata.generation_config = fetch_json(client, gen_url) or {}
    
    # Fetch tensor metadata
    safetensors_files = [f for f in metadata.files if f.endswith(".safetensors")]
    
    if safetensors_files:
        # Check for index file first
        if "model.safetensors.index.json" in metadata.files:
            index_url = f"{HF_BASE_URL}/{model_id}/resolve/main/model.safetensors.index.json"
            index_data = fetch_json(client, index_url)
            if index_data and "weight_map" in index_data:
                # Use weight_map as tensor list (simplified - just names and file mapping)
                for tensor_name, file_name in index_data["weight_map"].items():
                    metadata.tensor_map[tensor_name] = {"file": file_name}
        
        # If no index or need detailed info, fetch from first safetensors file
        if not metadata.tensor_map:
            # Try to get detailed tensor info from safetensors header
            for sf_file in safetensors_files[:3]:  # Limit to first 3 files
                url = f"{HF_BASE_URL}/{model_id}/resolve/main/{sf_file}"
                header = fetch_safetensors_header(client, url)
                if header:
                    for name, info in header.items():
                        if name != "__metadata__":
                            metadata.tensor_map[name] = {
                                "shape": tuple(info.get("shape", [])),
                                "dtype": info.get("dtype", "unknown"),
                                "file": sf_file,
                            }
    
    return metadata


def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dictionary with dot notation keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def compare_values(val_a: Any, val_b: Any) -> bool:
    """Compare two values, handling floating point precision."""
    if isinstance(val_a, float) and isinstance(val_b, float):
        return abs(val_a - val_b) < 1e-9
    return val_a == val_b


def diff_dicts(dict_a: Dict, dict_b: Dict, name: str = "config") -> Tuple[List, List, List]:
    """
    Compare two flattened dictionaries.
    Returns: (added, removed, modified) where each is a list of tuples
    """
    flat_a = flatten_dict(dict_a)
    flat_b = flatten_dict(dict_b)
    
    keys_a = set(flat_a.keys())
    keys_b = set(flat_b.keys())
    
    added = [(k, flat_b[k]) for k in sorted(keys_b - keys_a)]
    removed = [(k, flat_a[k]) for k in sorted(keys_a - keys_b)]
    
    modified = []
    for k in sorted(keys_a & keys_b):
        if not compare_values(flat_a[k], flat_b[k]):
            modified.append((k, flat_a[k], flat_b[k]))
    
    return added, removed, modified


def diff_tensors(map_a: Dict, map_b: Dict) -> Tuple[List, List, List]:
    """
    Compare tensor maps.
    Returns: (added, removed, modified)
    """
    keys_a = set(map_a.keys())
    keys_b = set(map_b.keys())
    
    added = sorted(keys_b - keys_a)
    removed = sorted(keys_a - keys_b)
    
    modified = []
    for k in sorted(keys_a & keys_b):
        info_a = map_a[k]
        info_b = map_b[k]
        
        shape_a = info_a.get("shape")
        shape_b = info_b.get("shape")
        dtype_a = info_a.get("dtype")
        dtype_b = info_b.get("dtype")
        
        if shape_a != shape_b or dtype_a != dtype_b:
            modified.append((k, info_a, info_b))
    
    return added, removed, modified


def summarize_layers(tensor_names: List[str]) -> Dict[str, int]:
    """Group tensors by layer prefix."""
    groups = {}
    for name in tensor_names:
        parts = name.split(".")
        if len(parts) >= 3 and parts[1] == "layers":
            prefix = f"{parts[0]}.{parts[1]}.{parts[2]}"
        elif len(parts) >= 2:
            prefix = f"{parts[0]}.{parts[1]}"
        else:
            prefix = parts[0]
        groups[prefix] = groups.get(prefix, 0) + 1
    return groups


def print_diff_report(meta_a: ModelMetadata, meta_b: ModelMetadata, verbose: bool = False):
    """Print formatted diff report."""
    c = Colors
    
    # Header
    print(f"\n{c.BOLD}{'='*70}{c.RESET}")
    print(f"{c.BOLD}HuggingFace Model Diff{c.RESET}")
    print(f"{'='*70}")
    print(f"\n{c.CYAN}Model A:{c.RESET} {meta_a.model_id}")
    print(f"{c.CYAN}Model B:{c.RESET} {meta_b.model_id}")
    
    # Check for errors
    if meta_a.error:
        print(f"\n{c.RED}Error fetching Model A: {meta_a.error}{c.RESET}")
        return
    if meta_b.error:
        print(f"\n{c.RED}Error fetching Model B: {meta_b.error}{c.RESET}")
        return
    
    # Quick summary
    print(f"\n{c.DIM}Files: {len(meta_a.files)} vs {len(meta_b.files)}{c.RESET}")
    print(f"{c.DIM}Tensors: {len(meta_a.tensor_map)} vs {len(meta_b.tensor_map)}{c.RESET}")
    
    # =========== CONFIG DIFF ===========
    print(f"\n{c.BOLD}{'─'*70}{c.RESET}")
    print(f"{c.BOLD}Configuration Diff (config.json){c.RESET}")
    print(f"{'─'*70}")
    
    added, removed, modified = diff_dicts(meta_a.config, meta_b.config)
    
    if not added and not removed and not modified:
        print(f"{c.GREEN}  Configurations are identical{c.RESET}")
    else:
        # Key differences summary
        key_params = ["model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
                     "num_key_value_heads", "vocab_size", "max_position_embeddings", 
                     "intermediate_size", "hidden_act"]
        
        print(f"\n{c.BOLD}Key Parameters:{c.RESET}")
        flat_a = flatten_dict(meta_a.config)
        flat_b = flatten_dict(meta_b.config)
        
        for param in key_params:
            val_a = flat_a.get(param, "—")
            val_b = flat_b.get(param, "—")
            if val_a != val_b:
                print(f"  {c.YELLOW}[~]{c.RESET} {param}: {c.RED}{val_a}{c.RESET} → {c.GREEN}{val_b}{c.RESET}")
            elif val_a != "—":
                print(f"  {c.DIM}[=]{c.RESET} {param}: {val_a}")
        
        if removed:
            print(f"\n{c.RED}Removed ({len(removed)}):{c.RESET}")
            for k, v in removed[:10] if not verbose else removed:
                print(f"  {c.RED}[-]{c.RESET} {k}: {v}")
            if len(removed) > 10 and not verbose:
                print(f"  {c.DIM}... and {len(removed) - 10} more (use --verbose){c.RESET}")
        
        if added:
            print(f"\n{c.GREEN}Added ({len(added)}):{c.RESET}")
            for k, v in added[:10] if not verbose else added:
                print(f"  {c.GREEN}[+]{c.RESET} {k}: {v}")
            if len(added) > 10 and not verbose:
                print(f"  {c.DIM}... and {len(added) - 10} more (use --verbose){c.RESET}")
        
        if modified and verbose:
            other_modified = [(k, a, b) for k, a, b in modified if k not in key_params]
            if other_modified:
                print(f"\n{c.YELLOW}Other Changes ({len(other_modified)}):{c.RESET}")
                for k, val_a, val_b in other_modified:
                    print(f"  {c.YELLOW}[~]{c.RESET} {k}: {val_a} → {val_b}")
    
    # =========== TENSOR DIFF ===========
    print(f"\n{c.BOLD}{'─'*70}{c.RESET}")
    print(f"{c.BOLD}Weight Structure Diff{c.RESET}")
    print(f"{'─'*70}")
    
    if not meta_a.tensor_map and not meta_b.tensor_map:
        print(f"  {c.YELLOW}No safetensors metadata available{c.RESET}")
    else:
        added_t, removed_t, modified_t = diff_tensors(meta_a.tensor_map, meta_b.tensor_map)
        
        if not added_t and not removed_t and not modified_t:
            print(f"{c.GREEN}  Weight structures are identical{c.RESET}")
        else:
            # Summary by layer groups
            if removed_t:
                removed_groups = summarize_layers(removed_t)
                print(f"\n{c.RED}Removed Tensors ({len(removed_t)}):{c.RESET}")
                for group, count in sorted(removed_groups.items()):
                    print(f"  {c.RED}[-]{c.RESET} {group}.* ({count} tensors)")
                if verbose:
                    for t in removed_t:
                        print(f"      {c.DIM}{t}{c.RESET}")
            
            if added_t:
                added_groups = summarize_layers(added_t)
                print(f"\n{c.GREEN}Added Tensors ({len(added_t)}):{c.RESET}")
                for group, count in sorted(added_groups.items()):
                    print(f"  {c.GREEN}[+]{c.RESET} {group}.* ({count} tensors)")
                if verbose:
                    for t in added_t:
                        print(f"      {c.DIM}{t}{c.RESET}")
            
            if modified_t:
                print(f"\n{c.YELLOW}Modified Tensors ({len(modified_t)}):{c.RESET}")
                for name, info_a, info_b in modified_t[:15] if not verbose else modified_t:
                    shape_a = info_a.get("shape", "?")
                    shape_b = info_b.get("shape", "?")
                    dtype_a = info_a.get("dtype", "?")
                    dtype_b = info_b.get("dtype", "?")
                    
                    changes = []
                    if shape_a != shape_b:
                        changes.append(f"shape: {shape_a} → {shape_b}")
                    if dtype_a != dtype_b:
                        changes.append(f"dtype: {dtype_a} → {dtype_b}")
                    
                    print(f"  {c.YELLOW}[~]{c.RESET} {name}")
                    for change in changes:
                        print(f"      {change}")
                
                if len(modified_t) > 15 and not verbose:
                    print(f"  {c.DIM}... and {len(modified_t) - 15} more (use --verbose){c.RESET}")
    
    # =========== COMPATIBILITY SUMMARY ===========
    print(f"\n{c.BOLD}{'─'*70}{c.RESET}")
    print(f"{c.BOLD}Compatibility Summary{c.RESET}")
    print(f"{'─'*70}")
    
    # Check architecture compatibility
    arch_a = meta_a.config.get("architectures", [meta_a.config.get("model_type", "unknown")])
    arch_b = meta_b.config.get("architectures", [meta_b.config.get("model_type", "unknown")])
    
    same_arch = arch_a == arch_b
    same_hidden = meta_a.config.get("hidden_size") == meta_b.config.get("hidden_size")
    same_layers = meta_a.config.get("num_hidden_layers") == meta_b.config.get("num_hidden_layers")
    same_vocab = meta_a.config.get("vocab_size") == meta_b.config.get("vocab_size")
    
    _, _, tensor_mods = diff_tensors(meta_a.tensor_map, meta_b.tensor_map)
    shape_changes = any(m[1].get("shape") != m[2].get("shape") for m in tensor_mods)
    
    print(f"  Architecture Match:     {'✓ Yes' if same_arch else '✗ No'} ({arch_a} vs {arch_b})")
    print(f"  Hidden Size Match:      {'✓ Yes' if same_hidden else '✗ No'}")
    print(f"  Layer Count Match:      {'✓ Yes' if same_layers else '✗ No'}")
    print(f"  Vocabulary Match:       {'✓ Yes' if same_vocab else '✗ No'}")
    print(f"  Weight Shapes Match:    {'✓ Yes' if not shape_changes else '✗ No'}")
    
    # Overall verdict
    if same_arch and same_hidden and same_layers and not shape_changes:
        print(f"\n  {c.GREEN}{c.BOLD}→ Models are structurally compatible (likely same base or fine-tune){c.RESET}")
    elif same_arch:
        print(f"\n  {c.YELLOW}{c.BOLD}→ Same architecture family but different configuration{c.RESET}")
    else:
        print(f"\n  {c.RED}{c.BOLD}→ Models have different architectures{c.RESET}")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two HuggingFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hf_diff.py gpt2 gpt2-medium
  python hf_diff.py meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf
  python hf_diff.py mistralai/Mistral-7B-v0.1 mistralai/Mistral-7B-Instruct-v0.2 --verbose

Environment Variables:
  HF_TOKEN: Set for accessing gated models
        """
    )
    
    parser.add_argument("model_a", help="First model ID (e.g., meta-llama/Llama-2-7b-hf)")
    parser.add_argument("model_b", help="Second model ID to compare against")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full detailed diff")
    parser.add_argument("--token", "-t", help="HuggingFace token for gated models")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if args.no_color:
        Colors.disable()
    
    print(f"Fetching metadata for {args.model_a} ...")
    print(f"Fetching metadata for {args.model_b} ...")
    
    with httpx.Client(headers=get_headers(args.token), timeout=60.0, follow_redirects=True) as client:
        meta_a = fetch_model_metadata(client, args.model_a)
        meta_b = fetch_model_metadata(client, args.model_b)
    
    if args.json:
        result = {
            "model_a": {
                "id": meta_a.model_id,
                "config": meta_a.config,
                "tensor_count": len(meta_a.tensor_map),
                "file_count": len(meta_a.files),
            },
            "model_b": {
                "id": meta_b.model_id,
                "config": meta_b.config,
                "tensor_count": len(meta_b.tensor_map),
                "file_count": len(meta_b.files),
            },
            "config_diff": {
                "added": diff_dicts(meta_a.config, meta_b.config)[0],
                "removed": diff_dicts(meta_a.config, meta_b.config)[1],
                "modified": [(k, a, b) for k, a, b in diff_dicts(meta_a.config, meta_b.config)[2]],
            },
            "tensor_diff": {
                "added": diff_tensors(meta_a.tensor_map, meta_b.tensor_map)[0],
                "removed": diff_tensors(meta_a.tensor_map, meta_b.tensor_map)[1],
                "modified_count": len(diff_tensors(meta_a.tensor_map, meta_b.tensor_map)[2]),
            }
        }
        print(json.dumps(result, indent=2, default=str))
    else:
        print_diff_report(meta_a, meta_b, verbose=args.verbose)


if __name__ == "__main__":
    main()
