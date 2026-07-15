"""
Master Script - Generate All Analysis Outputs
Run all analysis scripts and generate all visualizations
"""
import os
import sys
import subprocess
import time

# Scripts to run in order
SCRIPTS = [
    ('schematic_diagram.py', 'Pipeline Schematic'),
    ('data_stats.py', 'Data Statistics'),
    ('preprocessing_viz.py', 'Preprocessing Visualization'),
    ('feature_analysis.py', 'Feature Analysis'),
    ('feature_vs_accuracy.py', 'Feature vs Accuracy'),
    ('loop_analysis.py', 'Loop Analysis'),
    ('classifier_analysis.py', 'Classifier Analysis'),
]


def run_script(script_name, description):
    """Run a single analysis script."""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    print(f"\n{'='*60}")
    print(f" Running: {description}")
    print(f" Script: {script_name}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        
        if result.returncode == 0:
            print(result.stdout)
            elapsed = time.time() - start_time
            print(f"\n✓ {description} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"\n✗ Error in {script_name}:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"\n✗ Failed to run {script_name}: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("   KNSD Research Analysis - Generate All Outputs")
    print("="*60)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    total_start = time.time()
    success_count = 0
    failed = []
    
    for script_name, description in SCRIPTS:
        if run_script(script_name, description):
            success_count += 1
        else:
            failed.append(script_name)
    
    total_time = time.time() - total_start
    
    print("\n" + "="*60)
    print("   SUMMARY")
    print("="*60)
    print(f"\n  Total scripts run: {len(SCRIPTS)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total time: {total_time:.1f}s")
    
    if failed:
        print(f"\n  Failed scripts: {', '.join(failed)}")
    
    # List generated files
    print(f"\n  Output directory: {output_dir}")
    print("\n  Generated files:")
    
    if os.path.exists(output_dir):
        files = sorted(os.listdir(output_dir))
        for f in files:
            size = os.path.getsize(os.path.join(output_dir, f))
            print(f"    - {f} ({size/1024:.1f} KB)")
        print(f"\n  Total: {len(files)} files generated")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
