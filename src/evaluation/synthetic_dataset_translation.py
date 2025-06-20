import json
import os
from google.cloud import translate_v3
from google.oauth2 import service_account

INPUT_FILE = "../../data/processed/synthetic/synthetic_dataset_en.json"
OUTPUT_FILE = "../../data/processed/synthetic/synthetic_dataset_yakut.json"
GOOGLE_APPLICATION_CREDENTIALS = "REPLACE_WITH_YOUR_CREDENTIALS.json"

def translate_text(text, source_lang, target_lang, translate_client, project_id):
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
        return text  # fallback original text on error

def translate_dataset_incremental(dataset, translate_client, project_id, output_file, source_lang="en", target_lang="sah"):
    # Load already translated data if exists to resume
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, "r", encoding="utf-8") as f:
            translated_data = json.load(f)
        already_translated_count = len(translated_data)
    else:
        translated_data = []
        already_translated_count = 0

    total_items = len(dataset)

    for idx in range(already_translated_count, total_items):
        item = dataset[idx]
        print(f"Translating item {idx+1} / {total_items}")

        translated_item = {
            "id": item["id"],
            "context": translate_text(item["context"], source_lang, target_lang, translate_client, project_id),
            "question": translate_text(item["question"], source_lang, target_lang, translate_client, project_id),
            "options": [translate_text(opt, source_lang, target_lang, translate_client, project_id) for opt in item["options"]],
            "answer": translate_text(item["answer"], source_lang, target_lang, translate_client, project_id)
        }

        translated_data.append(translated_item)

        # Save progress after each item
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

    return translated_data


if __name__ == "__main__":
    # Load credentials and create client
    credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
    translate_client = translate_v3.TranslationServiceClient(credentials=credentials)
    project_id = credentials.project_id

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    translate_dataset_incremental(
        dataset,
        translate_client,
        project_id,
        OUTPUT_FILE,
        source_lang="en",
        target_lang="sah"
    )

    print("Translation complete!")
