#!/usr/bin/env python3
"""
Benchmark Report Generator for RF-DETR ONNX

Reads JSON benchmark results and generates a comprehensive markdown report
comparing Python and C++ implementations across different devices.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def load_benchmark_results(results_dir: str) -> Dict[str, Dict]:
    """Load all JSON benchmark results from the results directory.
    
    Args:
        results_dir: Path to directory containing JSON result files
        
    Returns:
        Dictionary mapping filename to parsed JSON data
    """
    results = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Warning: Results directory '{results_dir}' does not exist", file=sys.stderr)
        return results
    
    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                results[json_file.stem] = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse {json_file}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error reading {json_file}: {e}", file=sys.stderr)
    
    return results


def parse_result_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parse benchmark result filename into components.
    
    Expected format: {impl}_{device}_{test_type}_{model_name}
    e.g., python_cpu_images_rf-detr-nano.sim
    
    Args:
        filename: Result filename without extension
        
    Returns:
        Dictionary with impl, device, test_type, model_name or None if invalid
    """
    parts = filename.split('_')
    if len(parts) < 4:
        return None
    
    impl = parts[0]  # python or cpp
    device = parts[1]  # cpu or gpu
    # Everything between device and the last part could be test type or model
    # We need to find where test_type ends and model_name begins
    # Assume test_type is one word (e.g., 'images'), rest is model name
    test_type = parts[2]
    model_name = '_'.join(parts[3:])  # Rejoin in case model name has underscores
    
    return {
        'impl': impl,
        'device': device,
        'test_type': test_type,
        'model_name': model_name
    }


def extract_models(results: Dict[str, Dict]) -> List[str]:
    """Extract unique model names from result filenames.
    
    Args:
        results: Dictionary of benchmark results
        
    Returns:
        List of unique model names
    """
    models = set()
    for filename in results.keys():
        parsed = parse_result_filename(filename)
        if parsed:
            models.add(parsed['model_name'])
    
    return sorted(models)


def extract_test_types(results: Dict[str, Dict]) -> List[str]:
    """Extract unique test types from result filenames.
    
    Args:
        results: Dictionary of benchmark results
        
    Returns:
        List of unique test type names
    """
    test_types = set()
    for filename in results.keys():
        parsed = parse_result_filename(filename)
        if parsed:
            test_types.add(parsed['test_type'])
    
    return sorted(test_types)


def format_metric(value: float, is_best: bool, is_fps: bool = False) -> str:
    """Format a metric value with optional highlighting for best performance.
    
    Args:
        value: Numeric value to format
        is_best: Whether this is the best value (should be highlighted)
        is_fps: Whether this is FPS metric (higher is better)
        
    Returns:
        Formatted string with markdown bold if best
    """
    formatted = f"{value:.2f}"
    if is_best:
        emoji = " 🚀" if is_fps else ""
        return f"**{formatted}**{emoji}"
    return formatted


def generate_test_case_table(test_type: str, model_name: str, results: Dict[str, Dict]) -> str:
    """Generate markdown table for a specific test case and model.
    
    Args:
        test_type: Name of the test case (e.g., 'images')
        model_name: Name of the model
        results: All benchmark results
        
    Returns:
        Markdown formatted table string
    """
    # Collect rows for this test type and model
    rows = []
    implementations = ['python', 'cpp']
    devices = ['cpu', 'gpu']
    
    for impl in implementations:
        for dev in devices:
            key = f"{impl}_{dev}_{test_type}_{model_name}"
            if key in results:
                result = results[key]
                metrics = result.get('metrics', {})
                rows.append({
                    'impl': result.get('implementation', impl.title()),
                    'device': result.get('device', dev).upper(),
                    'preprocess': metrics.get('preprocessing', {}).get('mean', 0),
                    'ort_run': metrics.get('ort_run', {}).get('mean', 0),
                    'postprocess': metrics.get('postprocessing', {}).get('mean', 0),
                    'total': metrics.get('total_processing', {}).get('mean', 0),
                    'fps': metrics.get('total_processing', {}).get('fps', 0),
                })
    
    if not rows:
        return ""
    
    # Find best values (minimum for times, maximum for FPS)
    min_preprocess = min(r['preprocess'] for r in rows)
    min_ort_run = min(r['ort_run'] for r in rows)
    min_postprocess = min(r['postprocess'] for r in rows)
    min_total = min(r['total'] for r in rows)
    max_fps = max(r['fps'] for r in rows)
    
    # Build table
    table = f"### {test_type.replace('_', ' ').title()}\n\n"
    table += "| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |\n"
    table += "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |\n"
    
    for row in rows:
        pre_str = format_metric(row['preprocess'], row['preprocess'] == min_preprocess)
        ort_str = format_metric(row['ort_run'], row['ort_run'] == min_ort_run)
        post_str = format_metric(row['postprocess'], row['postprocess'] == min_postprocess)
        total_str = format_metric(row['total'], row['total'] == min_total)
        fps_str = format_metric(row['fps'], row['fps'] == max_fps, is_fps=True)
        
        table += f"| {row['impl']} | {row['device']} | {pre_str} | {ort_str} | {post_str} | {total_str} | {fps_str} |\n"
    
    table += "\n"
    return table


def generate_summary_section(results: Dict[str, Dict]) -> str:
    """Generate summary statistics section.
    
    Args:
        results: All benchmark results
        
    Returns:
        Markdown formatted summary string
    """
    if not results:
        return ""
    
    summary = "## Summary\n\n"
    
    # Count results by implementation and device
    python_count = sum(1 for k in results.keys() if k.startswith('python_'))
    cpp_count = sum(1 for k in results.keys() if k.startswith('cpp_'))
    cpu_count = sum(1 for k in results.keys() if '_cpu_' in k)
    gpu_count = sum(1 for k in results.keys() if '_gpu_' in k)
    
    summary += f"- **Total benchmarks:** {len(results)}\n"
    summary += f"- **Python benchmarks:** {python_count}\n"
    summary += f"- **C++ benchmarks:** {cpp_count}\n"
    summary += f"- **CPU benchmarks:** {cpu_count}\n"
    summary += f"- **GPU benchmarks:** {gpu_count}\n\n"
    
    # Extract iteration count from first result
    first_result = next(iter(results.values()))
    iterations = first_result.get('num_iterations', 'N/A')
    summary += f"- **Iterations per benchmark:** {iterations}\n\n"
    
    return summary


def generate_report(results_dir: str = "results", output_file: str = "results.md") -> None:
    """Generate comprehensive benchmark report from JSON results.
    
    Args:
        results_dir: Directory containing JSON result files
        output_file: Output markdown file path
    """
    # Load all results
    results = load_benchmark_results(results_dir)
    
    if not results:
        print("No benchmark results found. Cannot generate report.", file=sys.stderr)
        sys.exit(1)
    
    # Start building the markdown report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown = f"# RF-DETR ONNX Benchmark Results\n\n"
    markdown += f"**Generated:** {timestamp}\n\n"
    
    # Add summary section
    markdown += generate_summary_section(results)
    
    # Add separator
    markdown += "---\n\n"
    
    # Extract models and test types
    models = extract_models(results)
    test_types = extract_test_types(results)
    
    if not models:
        markdown += "*No models found in results.*\n"
    else:
        # Generate tables grouped by model
        for model_name in models:
            markdown += f"## Model: {model_name}\n\n"
            
            for test_type in test_types:
                table = generate_test_case_table(test_type, model_name, results)
                if table:
                    markdown += table
            
            # Add separator between models
            if model_name != models[-1]:
                markdown += "---\n\n"
    
    # Add footer
    markdown += "---\n\n"
    markdown += "*Note: Best performance values are **highlighted in bold**. " \
                "For timing metrics (ms), lower is better. For FPS, higher is better.*\n"
    
    # Write to file
    output_path = Path(output_file)
    output_path.write_text(markdown)
    print(f"✓ Report generated successfully: {output_file}")
    print(f"  - Total results processed: {len(results)}")
    print(f"  - Models: {len(models)}")
    print(f"  - Test cases: {len(test_types)}")


def main():
    """Main entry point for the report generator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate benchmark report from JSON results"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing JSON result files (default: results)"
    )
    parser.add_argument(
        "--output",
        default="results.md",
        help="Output markdown file (default: results.md)"
    )
    
    args = parser.parse_args()
    
    generate_report(args.results_dir, args.output)


if __name__ == "__main__":
    main()
