#!/usr/bin/env python3


import json
import argparse
import sys
import csv
from typing import List, Dict, Any, Union


def format_from_pairs(input_data: List[List[str]]) -> List[Dict[str, str]]:
    """
    Format from a list of [instruction, output] pairs.
    
    Args:
        input_data: List of [instruction, output] pairs
        
    Returns:
        List of dictionaries with "instruction" and "output" keys
    """
    formatted_data = []
    for pair in input_data:
        if len(pair) >= 2:
            formatted_data.append({
                "instruction": pair[0],
                "output": pair[1]
            })
    return formatted_data


def format_from_dict(input_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Format from a list of dictionaries, mapping various key names to "instruction" and "output".
    
    Args:
        input_data: List of dictionaries with various key names
        
    Returns:
        List of dictionaries with "instruction" and "output" keys
    """
    formatted_data = []
    
    # Common key mappings for instruction field
    instruction_keys = ['instruction', 'input', 'question', 'prompt', 'query', 'text']
    # Common key mappings for output field  
    output_keys = ['output', 'response', 'answer', 'reply', 'target', 'label']
    
    for item in input_data:
        instruction = None
        output = None
        
        # Find instruction field
        for key in instruction_keys:
            if key in item:
                instruction = str(item[key])
                break
                
        # Find output field
        for key in output_keys:
            if key in item:
                output = str(item[key])
                break
                
        if instruction and output:
            formatted_data.append({
                "instruction": instruction,
                "output": output
            })
        else:
            print(f"Warning: Could not find both instruction and output fields in item: {item}", file=sys.stderr)
    
    return formatted_data


def format_from_text_file(file_path: str, delimiter: str = '\t') -> List[Dict[str, str]]:
    """
    Format from a text file where each line contains instruction and output separated by delimiter.
    
    Args:
        file_path: Path to the text file
        delimiter: Delimiter separating instruction and output (default: tab)
        
    Returns:
        List of dictionaries with "instruction" and "output" keys
    """
    formatted_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(delimiter)
            if len(parts) >= 2:
                formatted_data.append({
                    "instruction": parts[0].strip(),
                    "output": parts[1].strip()
                })
            else:
                print(f"Warning: Line {line_num} does not have enough parts: {line}", file=sys.stderr)
    
    return formatted_data


def format_from_csv_file(file_path: str) -> List[Dict[str, str]]:
    """
    Format from a CSV file, automatically detecting instruction and output columns.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of dictionaries with "instruction" and "output" keys
    """
    formatted_data = []
    
    # Increase CSV field size limit for large fields
    csv.field_size_limit(sys.maxsize)
    
    # Common key mappings for instruction field
    instruction_keys = ['instruction', 'input', 'question', 'prompt', 'query', 'text']
    # Common key mappings for output field  
    output_keys = ['output', 'response', 'answer', 'reply', 'target', 'label']
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Try to detect delimiter
        sample = f.read(1024)
        f.seek(0)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        
        reader = csv.DictReader(f, delimiter=delimiter)
        
        # Find the appropriate column names
        instruction_col = None
        output_col = None
        
        for col in reader.fieldnames:
            if col.lower() in instruction_keys:
                instruction_col = col
            if col.lower() in output_keys:
                output_col = col
        
        if not instruction_col or not output_col:
            print(f"Available columns: {reader.fieldnames}", file=sys.stderr)
            raise ValueError(f"Could not find instruction column (looking for: {instruction_keys}) or output column (looking for: {output_keys})")
        
        print(f"Using '{instruction_col}' as instruction and '{output_col}' as output", file=sys.stderr)
        
        for row_num, row in enumerate(reader, 1):
            instruction = row.get(instruction_col, '').strip()
            output = row.get(output_col, '').strip()
            
            if instruction and output:
                formatted_data.append({
                    "instruction": instruction,
                    "output": output
                })
            else:
                print(f"Warning: Row {row_num} missing instruction or output", file=sys.stderr)
    
    return formatted_data


def load_input_data(input_file: str) -> Union[List[Dict[str, Any]], List[List[str]], str]:
    """
    Load input data from various file formats.
    
    Args:
        input_file: Path to input file
        
    Returns:
        Loaded data in appropriate format
    """
    if input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Assume it's a text file
        return input_file


def main():
    parser = argparse.ArgumentParser(
        description='Format input data into muri_tatoeba_sah.json structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Format from JSON file with different key names
  python format_to_muri_structure.py input.json -o output.json
  
  # Format from CSV file (auto-detects columns and delimiter)
  python format_to_muri_structure.py input.csv -o output.json --csv-mode
  
  # Format from tab-separated text file
  python format_to_muri_structure.py input.txt -o output.json --text-mode --delimiter "\\t"
  
  # Format from comma-separated text file
  python format_to_muri_structure.py input.csv -o output.json --text-mode --delimiter ","
        """
    )
    
    parser.add_argument('input_file', help='Input file to format')
    parser.add_argument('-o', '--output', help='Output JSON file (default: stdout)')
    parser.add_argument('--text-mode', action='store_true', 
                       help='Treat input as text file with delimited instruction/output pairs')
    parser.add_argument('--csv-mode', action='store_true', 
                       help='Treat input as CSV file (auto-detects delimiter and columns)')
    parser.add_argument('--delimiter', default='\t', 
                       help='Delimiter for text mode (default: tab)')
    parser.add_argument('--pretty', action='store_true', 
                       help='Pretty print JSON output')
    
    args = parser.parse_args()
    
    try:
        if args.csv_mode:
            # Format from CSV file
            formatted_data = format_from_csv_file(args.input_file)
        elif args.text_mode:
            # Format from text file
            formatted_data = format_from_text_file(args.input_file, args.delimiter)
        else:
            # Load and format from JSON
            input_data = load_input_data(args.input_file)
            
            if isinstance(input_data, list):
                if len(input_data) > 0:
                    if isinstance(input_data[0], dict):
                        # List of dictionaries
                        formatted_data = format_from_dict(input_data)
                    elif isinstance(input_data[0], list):
                        # List of lists (pairs)
                        formatted_data = format_from_pairs(input_data)
                    else:
                        raise ValueError("Unsupported input format: list of non-dict/non-list items")
                else:
                    formatted_data = []
            else:
                raise ValueError("Input must be a list")
        
        # Output the formatted data
        if args.pretty:
            json_output = json.dumps(formatted_data, indent=2, ensure_ascii=False)
        else:
            json_output = json.dumps(formatted_data, ensure_ascii=False)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"Formatted data written to {args.output}")
        else:
            print(json_output)
            
        print(f"Successfully formatted {len(formatted_data)} items", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 