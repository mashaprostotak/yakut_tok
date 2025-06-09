#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Main preprocessing script for the Yakut NLP project
# This script coordinates all preprocessing steps for both monolingual and parallel data

import argparse
import logging
import os
import sys
import importlib.util
from pathlib import Path

# Add the project root to the path to make imports work
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def check_module_exists(module_path):
    # Check if a module exists without importing it
    try:
        spec = importlib.util.find_spec(module_path)
        return spec is not None
    except ModuleNotFoundError:
        return False

def main():
    # Main entry point for preprocessing pipeline
    parser = argparse.ArgumentParser(description="Run preprocessing for Yakut NLP project")
    
    # Input/output directories
    parser.add_argument("--raw-dir", type=str, default="project/data/raw",
                        help="Directory containing raw data")
    parser.add_argument("--processed-dir", type=str, default="project/data/processed",
                        help="Directory to save processed data")
    
    # Data selection options
    parser.add_argument("--process-parallel", action="store_true",
                        help="Process parallel corpus data")
    parser.add_argument("--process-monolingual", action="store_true",
                        help="Process monolingual corpus data")
    parser.add_argument("--all", action="store_true",
                        help="Process all available data")
    
    # Parallel corpus options
    parser.add_argument("--parallel-corpus", type=str, choices=["tatoeba", "wikimedia", "all", "sample"], default="all",
                        help="Which parallel corpus to process")
    parser.add_argument("--src-lang", type=str, default="sah",
                        help="Source language code")
    parser.add_argument("--tgt-lang", type=str, default="en",
                        help="Target language code (English for translated Russian)")
    
    # Translation options
    parser.add_argument("--translate", action="store_true",
                        help="Translate Russian text to English using the neural model")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="Maximum number of lines to translate (for testing)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for translation")
    
    # Custom input files for testing
    parser.add_argument("--src-file", type=str, default=None,
                        help="Custom source file (for testing)")
    parser.add_argument("--tgt-file", type=str, default=None,
                        help="Custom target file (for testing)")
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="Custom output file prefix (for testing)")
    
    # Analysis options
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip analysis steps")
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(args.processed_dir, exist_ok=True)
    parallel_dir = Path(args.processed_dir) / "parallel"
    mono_dir = Path(args.processed_dir) / "monolingual"
    os.makedirs(parallel_dir, exist_ok=True)
    os.makedirs(mono_dir, exist_ok=True)
    
    # Process parallel data
    if args.process_parallel or args.all:
        logger.info("Processing parallel corpus data")
        
        from project.src.preprocessing.process_parallel import main as process_parallel_main
        
        # Build arguments for parallel processing
        parallel_args = [
            "--input-dir", f"{args.raw_dir}/opus",
            "--output-dir", str(parallel_dir),
            "--corpus", args.parallel_corpus,
            "--src-lang", args.src_lang,
            "--tgt-lang", args.tgt_lang
        ]
        
        if args.translate:
            parallel_args.append("--translate")
        
        if args.max_lines:
            parallel_args.extend(["--max-lines", str(args.max_lines)])
            
        if args.batch_size:
            parallel_args.extend(["--batch-size", str(args.batch_size)])
            
        # Add custom files if provided
        if args.src_file:
            parallel_args.extend(["--src-file", args.src_file])
        
        if args.tgt_file:
            parallel_args.extend(["--tgt-file", args.tgt_file])
            
        if args.output_prefix:
            parallel_args.extend(["--output-prefix", args.output_prefix])
        
        # Call the parallel processing script
        import sys
        old_argv = sys.argv
        sys.argv = ["process_parallel.py"] + parallel_args
        process_parallel_main()
        sys.argv = old_argv
        
        logger.info("Parallel corpus processing completed")
    
    # Process monolingual data
    if args.process_monolingual or args.all:
        logger.info("Processing monolingual corpus data")
        
        # Check if the module exists
        if check_module_exists("project.src.preprocessing.process_oscar"):
            from project.src.preprocessing.process_oscar import main as process_oscar_main
            
            # Build arguments for monolingual processing
            mono_args = [
                "--input-dir", f"{args.raw_dir}/oscar",
                "--output-dir", str(mono_dir),
                "--lang", args.src_lang
            ]
            
            # Call the monolingual processing script
            import sys
            old_argv = sys.argv
            sys.argv = ["process_oscar.py"] + mono_args
            process_oscar_main()
            sys.argv = old_argv
            
            logger.info("Monolingual corpus processing completed")
        else:
            logger.warning("Monolingual processing module not found. Skipping.")
    
    # Analyze the processed data
    if not args.skip_analysis and (args.process_parallel or args.process_monolingual or args.all):
        logger.info("Running analysis on processed data")
        
        # Create analysis directory
        analysis_dir = Path(args.processed_dir) / "analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        
        if args.process_monolingual or args.all:
            # Check if the module exists
            if check_module_exists("project.src.preprocessing.analysis.analyze_monolingual_data"):
                from project.src.preprocessing.analysis.analyze_monolingual_data import main as analyze_mono_main
                
                # Build arguments for monolingual analysis
                analyze_mono_args = [
                    "--input-dir", str(mono_dir),
                    "--output-dir", str(analysis_dir),
                    "--lang", args.src_lang
                ]
                
                # Call the monolingual analysis script
                import sys
                old_argv = sys.argv
                sys.argv = ["analyze_monolingual_data.py"] + analyze_mono_args
                analyze_mono_main()
                sys.argv = old_argv
                
                logger.info("Monolingual data analysis completed")
            else:
                logger.warning("Monolingual analysis module not found. Skipping.")
        
        if args.process_parallel or args.all:
            # Check if the module exists
            if check_module_exists("project.src.preprocessing.analysis.analyze_parallel_data"):
                from project.src.preprocessing.analysis.analyze_parallel_data import main as analyze_parallel_main
                
                # Build arguments for parallel analysis
                analyze_parallel_args = [
                    "--input-dir", str(parallel_dir),
                    "--output-dir", str(analysis_dir),
                    "--src-lang", args.src_lang,
                    "--tgt-lang", args.tgt_lang
                ]
                
                # Call the parallel analysis script
                import sys
                old_argv = sys.argv
                sys.argv = ["analyze_parallel_data.py"] + analyze_parallel_args
                analyze_parallel_main()
                sys.argv = old_argv
                
                logger.info("Parallel data analysis completed")
            else:
                logger.warning("Parallel analysis module not found. Skipping.")
        
        logger.info("Data analysis steps completed")
    elif args.skip_analysis:
        logger.info("Analysis steps skipped as requested")
    
    logger.info("All preprocessing steps completed successfully")

if __name__ == "__main__":
    main()
