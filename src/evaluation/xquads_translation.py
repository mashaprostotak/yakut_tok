import json
import os
from google.cloud import translate_v3
from google.oauth2 import service_account
from datasets import load_dataset

INPUT_PATH_DEEPMIND = "../../data/raw/xquad/xquad.en.json"
OUTPUT_PATH_HUGGINGFACE = "../../data/processed/xquad/xquad_translated_sah_huggingface.json"
OUTPUT_PATH_DEEPMIND = "../../data/processed/xquad/xquad_translated_sah_deepmind.json"
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
        return text  # Return original text on error

def translate_xquad_incremental(xquad_data, translate_client, project_id, output_file, source_lang="en", target_lang="ru"):
    translated_data = {"data": []}

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, "r", encoding="utf-8") as f:
            translated_data = json.load(f)
        already_translated_count = len(translated_data.get("data", []))
    else:
        already_translated_count = 0

    total_articles = len(xquad_data["data"])

    for idx, article in enumerate(xquad_data["data"]):
        if idx < already_translated_count:
            continue

        translated_article = {"title": article.get("title", ""), "paragraphs": []}
        for paragraph in article["paragraphs"]:
            translated_context = translate_text(paragraph["context"], source_lang, target_lang, translate_client, project_id)

            translated_paragraph = {"context": translated_context, "qas": []}
            for qa in paragraph["qas"]:
                translated_question = translate_text(qa["question"], source_lang, target_lang, translate_client, project_id)

                translated_answers = []
                for ans in qa.get("answers", []):
                    translated_answer_text = translate_text(ans["text"], source_lang, target_lang, translate_client, project_id)
                    translated_answers.append({
                        "text": translated_answer_text,
                        "answer_start": -1  # Not valid after translation
                    })

                translated_paragraph["qas"].append({
                    "id": qa["id"],
                    "question": translated_question,
                    "answers": translated_answers
                })
            translated_article["paragraphs"].append(translated_paragraph)

        translated_data["data"].append(translated_article)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

        print(f"Translated and saved article {idx + 1} / {total_articles}")

    return translated_data

def convert_xquad_hf_to_squad_format(hf_dataset):
    articles = []
    for example in hf_dataset:
        article = {
            "title": example.get("id", ""),
            "paragraphs": [
                {
                    "context": example["context"],
                    "qas": [
                        {
                            "id": example["id"],
                            "question": example["question"],
                            "answers": [
                                {"text": t, "answer_start": s}
                                for t, s in zip(example["answers"]["text"], example["answers"]["answer_start"])
                            ]
                        }
                    ]
                }
            ]
        }
        articles.append(article)
    return {"data": articles}

if __name__ == "__main__":
    # Load credentials and create client
    credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
    translate_client = translate_v3.TranslationServiceClient(credentials=credentials)
    project_id = credentials.project_id

    # Load and convert the DeepMind XQuAD dataset
    with open(INPUT_PATH_DEEPMIND, "r", encoding="utf-8") as f:
        xquad_deepmind = json.load(f)

    xquad_deepmind = convert_xquad_hf_to_squad_format(xquad_deepmind)

    translate_xquad_incremental(
        xquad_deepmind,
        translate_client,
        project_id,
        OUTPUT_PATH_DEEPMIND,
        source_lang="en",
        target_lang="sah"
    )

    print("DeepMind XQuAD done!")

    # Load and convert the latest XQuAD dataset from Hugging Face
    dataset = load_dataset("xquad", "xquad.en")
    xquad_huggingface = convert_xquad_hf_to_squad_format(dataset["validation"])

    translate_xquad_incremental(
        xquad_huggingface,
        translate_client,
        project_id,
        OUTPUT_PATH_HUGGINGFACE,
        source_lang="en",
        target_lang="sah"
    )

    print("HuggingFace XQuAD done!")
