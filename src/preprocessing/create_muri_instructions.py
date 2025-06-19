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

try:
    from google.cloud import translate_v3
    from google.oauth2 import service_account
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_API_AVAILABLE = True
except ImportError:
    GEMINI_API_AVAILABLE = False

def read_parallel_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_russian_with_razdel(text):
    tokens = list(razdel.tokenize(text))
    
    def smart_reconstruct(original_text, token_objects):
        reconstructed = ""
        original_pos = 0
        
        for token_obj in token_objects:
            token_text = token_obj.text
            token_pos = original_text.find(token_text, original_pos)
            if token_pos >= 0:
                reconstructed += original_text[original_pos:token_pos]
                reconstructed += token_text
                original_pos = token_pos + len(token_text)
        
        reconstructed += original_text[original_pos:]
        return reconstructed
    
    processed_text = smart_reconstruct(text, tokens)
    return processed_text

def generate_reverse_instructions(text_list, model_name="gemini-2.0-flash-lite"):
    few_shot_prompt = """Answer: Apache Kafka is a distributed system. The main components of Apache Kafka [...]
> What kind of instruction could this be the answer to?
Instruction: What are the main components of Apache Kafka?

Answer: [DOC]
> What kind of instruction could this be the answer to?
Instruction:"""
    
    instructions = []

    if model_name.startswith("gemini"):
        if not GEMINI_API_AVAILABLE:
            return ["" for _ in text_list]
            
        api_key = os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        for text in tqdm(text_list, desc="Generating reverse instructions with Gemini"):
            truncated_text = text[:15000]
            prompt = few_shot_prompt.replace("[DOC]", truncated_text)
            
            generation_config = genai.types.GenerationConfig(temperature=0.7)
            response = model.generate_content(prompt, generation_config=generation_config)
            instruction = response.text if response.text else ""
            
            instructions.append(instruction.strip())
        return instructions

    else:
        model_tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        for text in tqdm(text_list, desc=f"Generating reverse instructions with {model_name}"):
            truncated_text = text[:500]
            processed_text = preprocess_russian_with_razdel(truncated_text)
            prompt = few_shot_prompt.replace("[DOC]", processed_text)
            
            inputs = model_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(
                inputs.input_ids,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            instruction = model_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if any(char in instruction for char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'):
                instruction = preprocess_russian_with_razdel(instruction)
                
            instructions.append(instruction)
        return instructions

def translate_text(text, source_lang, target_lang, translate_client, project_id):
    if not translate_client or not project_id:
        return text

    parent = f"projects/{project_id}/locations/global"
    
    response = translate_client.translate_text(
        contents=[text],
        parent=parent,
        mime_type="text/plain",
        source_language_code=source_lang,
        target_language_code=target_lang,
    )
    
    return response.translations[0].translated_text

def create_muri_instructions(parallel_dir, output_dir, model_name, dataset_name, english_data, russian_data, yakut_texts, args):
    results = {}
    
    translate_client = None
    project_id = None
    
    if GOOGLE_CLOUD_AVAILABLE:
        credentials_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if credentials_json_str:
            credentials_info = json.loads(credentials_json_str)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            translate_client = translate_v3.TranslationServiceClient(credentials=credentials)
            project_id = credentials_info.get("project_id")
        
        

       
    
    # MURI Pipeline (Instructions Only)
        
    # 1. Generate instructions in English using the specified model
    print(f"Generating instructions in English using {model_name}...")
    eng_instructions = generate_reverse_instructions(english_data, model_name=model_name)
    
    # 2. Translate English instructions to Russian
    print("Translating generated instructions to Russian...")
    ru_instructions = []
    for instr in tqdm(eng_instructions, desc="Translating instructions to Russian"):
        if instr:
            translated_instr = translate_text(instr, "en", "ru", translate_client, project_id)
        else:
            translated_instr = ""
        ru_instructions.append(translated_instr)

    # 3. Translate Russian instructions to Yakut
    print("Translating Russian instructions to Yakut...")
    yakut_instructions = []
    for instr in tqdm(ru_instructions, desc="Translating instructions to Yakut"):
        if instr:
            instr=instr.removeprefix("Инструкция: ")
            translated_instr = translate_text(instr, "ru", "sah", translate_client, project_id)
        else:
            translated_instr = ""
        yakut_instructions.append(translated_instr)
        
    # End MURI Pipeline
    
    ru_pairs = []
    for i in range(len(russian_data)):
        ru_pairs.append({
            "instruction": ru_instructions[i],
            "output": russian_data[i]
        })
    
    yakut_pairs = []
    for i in range(len(yakut_texts)):
        yakut_pairs.append({
            "instruction": yakut_instructions[i],
            "output": yakut_texts[i]
        })
    
    ru_output_file = os.path.join(output_dir, f"muri_{dataset_name}_ru.json")
    if not args.output_file:
        with open(ru_output_file, 'w', encoding='utf-8') as f:
            json.dump(ru_pairs, f, ensure_ascii=False, indent=2)
    
    yakut_output_file = os.path.join(output_dir, f"muri_{dataset_name}_sah.json")
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(yakut_pairs, f, ensure_ascii=False, indent=2)
    else:
        with open(yakut_output_file, 'w', encoding='utf-8') as f:
            json.dump(yakut_pairs, f, ensure_ascii=False, indent=2)
    
    return {"ru": ru_pairs, "sah": yakut_pairs}

if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Generate MURI instruction-output pairs optimized for Russian text")
    parser.add_argument("--parallel_dir", type=str, default="project/data/processed/parallel")
    parser.add_argument("--output_dir", type=str, default="project/data/muri_instructions")
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL_NAME", "gemini-1.5-flash-latest"))
    parser.add_argument("--datasets", type=str, default="tatoeba,wikimedia")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    datasets_to_process = [name.strip() for name in args.datasets.split(',')]
    
    for dataset in datasets_to_process:
        full_english_data = read_parallel_data(os.path.join(args.parallel_dir, f"{dataset}_new.en"))
        full_russian_data = read_parallel_data(os.path.join(args.parallel_dir, f"{dataset}.ru"))
        full_yakut_texts = read_parallel_data(os.path.join(args.parallel_dir, f"{dataset}.sah"))

        start_index = args.start_index
        limit = args.limit
        
        end_index = None
        if limit is not None:
            end_index = start_index + limit
            
        english_data_slice = full_english_data[start_index:end_index]
        russian_data_slice = full_russian_data[start_index:end_index]
        yakut_texts_slice = full_yakut_texts[start_index:end_index]
        
        if english_data_slice:
            create_muri_instructions(
                args.parallel_dir,
                args.output_dir,
                model_name=args.model,
                dataset_name=dataset,
                english_data=english_data_slice,
                russian_data=russian_data_slice,
                yakut_texts=yakut_texts_slice,
                args=args
            ) 