import argparse
import os
import sys
import importlib.util
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

def check_module_exists(module_path):
    try:
        spec = importlib.util.find_spec(module_path)
        return spec is not None
    except ModuleNotFoundError:
        return False

def main():
    parser = argparse.ArgumentParser(description="Run preprocessing for Yakut NLP project")
    
    parser.add_argument("--raw-dir", type=str, default="project/data/raw")
    parser.add_argument("--processed-dir", type=str, default="project/data/processed")
    
    parser.add_argument("--process-parallel", action="store_true")
    parser.add_argument("--process-monolingual", action="store_true")
    parser.add_argument("--all", action="store_true")
    
    parser.add_argument("--parallel-corpus", type=str, choices=["tatoeba", "wikimedia", "all", "sample"], default="all")
    parser.add_argument("--src-lang", type=str, default="sah")
    parser.add_argument("--tgt-lang", type=str, default="en")
    
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    
    parser.add_argument("--src-file", type=str, default=None)
    parser.add_argument("--tgt-file", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    
    parser.add_argument("--skip-analysis", action="store_true")
    
    args = parser.parse_args()
    
    os.makedirs(args.processed_dir, exist_ok=True)
    parallel_dir = Path(args.processed_dir) / "parallel"
    mono_dir = Path(args.processed_dir) / "monolingual"
    os.makedirs(parallel_dir, exist_ok=True)
    os.makedirs(mono_dir, exist_ok=True)
    
    if args.process_parallel or args.all:
        from project.src.preprocessing.process_parallel import main as process_parallel_main
        
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
            
        if args.src_file:
            parallel_args.extend(["--src-file", args.src_file])
        
        if args.tgt_file:
            parallel_args.extend(["--tgt-file", args.tgt_file])
            
        if args.output_prefix:
            parallel_args.extend(["--output-prefix", args.output_prefix])
        
        old_argv = sys.argv
        sys.argv = ["process_parallel.py"] + parallel_args
        process_parallel_main()
        sys.argv = old_argv
    
    if args.process_monolingual or args.all:
        if check_module_exists("project.src.preprocessing.process_oscar"):
            from project.src.preprocessing.process_oscar import main as process_oscar_main
            
            mono_args = [
                "--input-dir", f"{args.raw_dir}/oscar",
                "--output-dir", str(mono_dir),
                "--lang", args.src_lang
            ]
            
            old_argv = sys.argv
            sys.argv = ["process_oscar.py"] + mono_args
            process_oscar_main()
            sys.argv = old_argv
    
    if not args.skip_analysis and (args.process_parallel or args.process_monolingual or args.all):
        analysis_dir = Path(args.processed_dir) / "analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        
        if args.process_monolingual or args.all:
            if check_module_exists("project.src.preprocessing.analysis.analyze_monolingual_data"):
                from project.src.preprocessing.analysis.analyze_monolingual_data import main as analyze_mono_main
                
                analyze_mono_args = [
                    "--input-dir", str(mono_dir),
                    "--output-dir", str(analysis_dir),
                    "--lang", args.src_lang
                ]
                
                old_argv = sys.argv
                sys.argv = ["analyze_monolingual_data.py"] + analyze_mono_args
                analyze_mono_main()
                sys.argv = old_argv
        
        if args.process_parallel or args.all:
            if check_module_exists("project.src.preprocessing.analysis.analyze_parallel_data"):
                from project.src.preprocessing.analysis.analyze_parallel_data import main as analyze_parallel_main
                
                analyze_parallel_args = [
                    "--input-dir", str(parallel_dir),
                    "--output-dir", str(analysis_dir),
                    "--src-lang", args.src_lang,
                    "--tgt-lang", args.tgt_lang
                ]
                
                old_argv = sys.argv
                sys.argv = ["analyze_parallel_data.py"] + analyze_parallel_args
                analyze_parallel_main()
                sys.argv = old_argv

if __name__ == "__main__":
    main()
