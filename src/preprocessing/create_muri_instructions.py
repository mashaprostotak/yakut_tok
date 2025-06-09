#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to implement the MURI (Multilingual Reverse Instructions) methodology
as described in the paper: https://arxiv.org/pdf/2409.12958

"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from dotenv import load_dotenv
import razdel

# Conditional Google Cloud & Gemini import
try:
    from google.cloud import translate_v3
    from google.oauth2 import service_account
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    print("Google Cloud Translation not available")

try:
    import google.generativeai as genai
    GEMINI_API_AVAILABLE = True
except ImportError:
    GEMINI_API_AVAILABLE = False
    print("Google Generative AI not available")

# Function to read parallel data
def read_parallel_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

# Function to preprocess Russian text with razdel
def preprocess_russian_with_razdel(text):
    # Use razdel to properly tokenize and clean Russian text with smart reconstruction
    # Use razdel for proper Russian tokenization
    tokens = list(razdel.tokenize(text))
    
    # Smart reconstruction that preserves original spacing
    def smart_reconstruct(original_text, token_objects):
        # Reconstruct text preserving original spacing patterns
        reconstructed = ""
        original_pos = 0
        
        for token_obj in token_objects:
            token_text = token_obj.text
            # Find token in original text starting from current position
            token_pos = original_text.find(token_text, original_pos)
            if token_pos >= 0:
                # Add any whitespace/characters before the token
                reconstructed += original_text[original_pos:token_pos]
                # Add the token
                reconstructed += token_text
                # Update position
                original_pos = token_pos + len(token_text)
        
        # Add any remaining characters
        reconstructed += original_text[original_pos:]
        return reconstructed
    
    # Use smart reconstruction to preserve spacing
    processed_text = smart_reconstruct(text, tokens)
    
    return processed_text

# Function to generate reverse instructions using Transformers or Gemini
def generate_reverse_instructions(text_list, model_name="gemini-2.0-flash-lite"):
    # Generates reverse instructions for a list of texts
    # It checks the model_name to decide whether to use the Gemini API
    # or a local Hugging Face Transformers model
    
    few_shot_prompt = """Answer: Apache Kafka is a distributed system. The main components of Apache Kafka [...]
> What kind of instruction could this be the answer to?
Instruction: What are the main components of Apache Kafka?

Answer: [DOC]
> What kind of instruction could this be the answer to?
Instruction:"""
    
    instructions = []

    # Use Gemini API if model_name starts with "gemini"
    if model_name.startswith("gemini"):
        if not GEMINI_API_AVAILABLE:
            print("Gemini SDK not installed. Cannot proceed with Gemini model.")
            return ["" for _ in text_list]
            
        print(f"Using Gemini model for instruction generation: {model_name}")
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            print("Gemini model initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize Gemini model: {e}")
            return ["" for _ in text_list]

        for text in tqdm(text_list, desc="Generating reverse instructions with Gemini"):
            # Use a larger context limit for Gemini
            truncated_text = text[:15000]
            prompt = few_shot_prompt.replace("[DOC]", truncated_text)
            
            try:
                # Add extra safety settings for the generation
                generation_config = genai.types.GenerationConfig(
                    temperature=0.7,
                )
                response = model.generate_content(prompt, generation_config=generation_config)
                instruction = response.text
            except Exception as e:
                print(f"Error generating instruction for text: '{text[:50]}...'. Error: {e}")
                instruction = ""
            
            instructions.append(instruction.strip())
        return instructions

    # Fallback to local Hugging Face Transformers model
    else:
        print(f"Loading local model for instruction generation: {model_name}")
        try:
            model_tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print(f"Model loaded successfully: {model_name}")
        except Exception as e:
            print(f"Failed to load local model: {e}")
            return ["" for _ in text_list]

        for text in tqdm(text_list, desc=f"Generating reverse instructions with {model_name}"):
            truncated_text = text[:500]
            processed_text = preprocess_russian_with_razdel(truncated_text)
            prompt = few_shot_prompt.replace("[DOC]", processed_text)
            
            inputs = model_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(
                inputs.input_ids, 
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            instruction = model_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if any(char in instruction for char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'):
                instruction = preprocess_russian_with_razdel(instruction)
                
            instructions.append(instruction)
        return instructions

# Function to translate text using Google Cloud Translation API
def translate_text(text, source_lang, target_lang, translate_client, project_id):
    # Translate text using a pre-initialized Google Cloud Translation client
    if not translate_client or not project_id:
        print(f"Translation client not available")
        return text

    try:
        parent = f"projects/{project_id}/locations/global"
        
        response = translate_client.translate_text(
            contents=[text],
            parent=parent,
            mime_type="text/plain",
            source_language_code=source_lang,
            target_language_code=target_lang,
        )
        
        return response.translations[0].translated_text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text  # Return original text on error

# Main function to process data and create MURI instructions
def create_muri_instructions(parallel_dir, output_dir, model_name, dataset_name, english_data, russian_data, yakut_texts, args):
    results = {}
    
    # Google Cloud Translation Client Initialization
    translate_client = None
    project_id = None
    
    if GOOGLE_CLOUD_AVAILABLE:
        try:
            # Priority 1: Use raw JSON from environment variable
            credentials_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
            if credentials_json_str:
                try:
                    # Ensure the string is parsed correctly, even with escaped newlines
                    credentials_info = json.loads(credentials_json_str)
                    credentials = service_account.Credentials.from_service_account_info(credentials_info)
                    translate_client = translate_v3.TranslationServiceClient(credentials=credentials)
                    project_id = credentials_info.get("project_id")
                    print("Initialized Google Cloud client from JSON environment variable.")
                except json.JSONDecodeError as e:
                    print(f"ERROR: Failed to parse GOOGLE_CREDENTIALS_JSON.")
                    print(f"   Please ensure it's a valid JSON string, enclosed in single quotes in your .env file.")
                    print(f"   JSON parser error: {e}")
                    translate_client = None
            
            # Priority 2: Fallback to file path from environment variable
            elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                # The client will find the credentials via the env var automatically
                translate_client = translate_v3.TranslationServiceClient()
                # We still need to extract project_id for the API calls
                credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                if credentials_path and os.path.exists(credentials_path):
                    with open(credentials_path, 'r', encoding='utf-8') as f:
                        credentials_info = json.load(f)
                        project_id = credentials_info.get("project_id")
                else: # Fallback for project_id if file is not readable but client initialized
                    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")

                if project_id:
                    print("Initialized Google Cloud client from credentials file path.")
                else:
                    translate_client = None # Could not get project_id

            if not translate_client:
                 print("Could not initialize Google Cloud ")

        except Exception as e:
            print(f"Failed to initialize Google Cloud client: {e}")
            translate_client = None # Ensure client is None on failure
            project_id = None
    else:
        print("Warning: Google Cloud Translation not available. Translations will be skipped.")
    
    # MURI Pipeline (Instructions Only)
        
    # 1. Generate instructions in English using the specified model
    print(f"Generating instructions in English using {model_name}...")
    eng_instructions = generate_reverse_instructions(english_data, model_name=model_name)
    
    # 2. Translate English instructions to Russian
    print("Translating generated instructions to Russian...")
    ru_instructions = []
    for instr in tqdm(eng_instructions, desc="Translating instructions to Russian"):
        if instr: # Only translate non-empty strings
            translated_instr = translate_text(instr, "en", "ru", translate_client, project_id)
        else:
            translated_instr = ""
        ru_instructions.append(translated_instr)

    # 3. Translate Russian instructions to Yakut
    print("Translating Russian instructions to Yakut...")
    yakut_instructions = []
    for instr in tqdm(ru_instructions, desc="Translating instructions to Yakut"):
        if instr: # Only translate non-empty strings
            instr=instr.removeprefix("Инструкция: ")
            translated_instr = translate_text(instr, "ru", "sah", translate_client, project_id)
        else:
            translated_instr = ""
        yakut_instructions.append(translated_instr)
        
    # End MURI Pipeline
    
    # Create Russian instruction-output pairs using original Russian text
    ru_pairs = []
    for i in range(len(russian_data)):
        ru_pairs.append({
            "instruction": ru_instructions[i],
            "output": russian_data[i]
        })
    
    # Create Yakut instruction-output pairs using original Yakut text
    yakut_pairs = []
    for i in range(len(yakut_texts)):
        yakut_pairs.append({
            "instruction": yakut_instructions[i],
            "output": yakut_texts[i]
        })
    
    # Save Russian results
    ru_output_file = os.path.join(output_dir, f"muri_{dataset_name}_ru.json")
    if not args.output_file: # Use default if not specified
        with open(ru_output_file, 'w', encoding='utf-8') as f:
            json.dump(ru_pairs, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(ru_pairs)} Russian instruction-output pairs to {ru_output_file}")
    
    # Save Yakut results
    yakut_output_file = os.path.join(output_dir, f"muri_{dataset_name}_sah.json")
    if args.output_file:
        # If a specific output file is given, save only the Yakut pairs there
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(yakut_pairs, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(yakut_pairs)} Yakut instruction-output pairs to {args.output_file}")
    else:
        with open(yakut_output_file, 'w', encoding='utf-8') as f:
            json.dump(yakut_pairs, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(yakut_pairs)} Yakut instruction-output pairs to {yakut_output_file}")
    
    return {"ru": ru_pairs, "sah": yakut_pairs}

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Generate MURI instruction-output pairs optimized for Russian text")
    parser.add_argument("--parallel_dir", type=str, default="project/data/processed/parallel",
                        help="Directory containing parallel data")
    parser.add_argument("--output_dir", type=str, default="project/data/muri_instructions",
                        help="Directory to save MURI instructions")
    parser.add_argument("--model", type=str, 
                        default=os.environ.get("MODEL_NAME", "gemini-1.5-flash-latest"),
                        help="Model to use for instruction generation (e.g., gemini-1.5-flash-latest, google/flan-t5-xl).")
    parser.add_argument("--datasets", type=str, default="tatoeba,wikimedia", 
                        help="Comma-separated list of datasets to process.")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Index to start processing from.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of lines to process.")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Specific output file to save results (primarily for Yakut).")
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print("Starting MURI pipeline")
    
    datasets_to_process = [name.strip() for name in args.datasets.split(',')]
    
    for dataset in datasets_to_process:
        print(f"\n----- Processing dataset: {dataset} -----")
        
        # Read the full data first
        try:
            full_english_data = read_parallel_data(os.path.join(args.parallel_dir, f"{dataset}_new.en"))
            full_russian_data = read_parallel_data(os.path.join(args.parallel_dir, f"{dataset}.ru"))
            full_yakut_texts = read_parallel_data(os.path.join(args.parallel_dir, f"{dataset}.sah"))
        except FileNotFoundError as e:
            print(f"Error loading data for {dataset}: {e}")
            continue

        # Determine the slice to process
        start_index = args.start_index
        limit = args.limit
        
        end_index = None
        if limit is not None:
            end_index = start_index + limit
            
        english_data_slice = full_english_data[start_index:end_index]
        russian_data_slice = full_russian_data[start_index:end_index]
        yakut_texts_slice = full_yakut_texts[start_index:end_index]
        
        if not english_data_slice:
            print(f"No data to process for {dataset} with the given start-index and limit.")
            continue

        print(f"Processing {len(english_data_slice)} lines for dataset '{dataset}' starting at index {start_index}.")

        create_muri_instructions(
            args.parallel_dir,
            args.output_dir,
            model_name=args.model,
            dataset_name=dataset,
            # Pass sliced data to the function
            english_data=english_data_slice,
            russian_data=russian_data_slice,
            yakut_texts=yakut_texts_slice,
            args=args
        ) 