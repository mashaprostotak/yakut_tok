#!/usr/bin/env python3
import sys
import os

class Logger:
    def __init__(self, output_dir):
        self.terminal = sys.stdout
        self.log_file = open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def main():
    # Set up output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up logging
    sys.stdout = Logger(output_dir)
    
    # Import and run the analysis
    import analyze_parallel_data
    analyze_parallel_data.main()
    
    # Close the log file
    sys.stdout.log_file.close()
    sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main() 