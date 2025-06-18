import json

def check_uniqueness(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    contexts = set()
    questions = set()

    duplicate_contexts = []
    duplicate_questions = []

    for i, item in enumerate(data):
        context = item.get("context", "").strip()
        question = item.get("question", "").strip()

        if context in contexts:
            duplicate_contexts.append((i, context))
        else:
            contexts.add(context)

        if question in questions:
            duplicate_questions.append((i, question))
        else:
            questions.add(question)

    print(f"\nTotal entries: {len(data)}")
    print(f"Unique contexts: {len(contexts)}")
    print(f"Unique questions: {len(questions)}")

    if duplicate_contexts:
        print(f"\nFound {len(duplicate_contexts)} duplicate context(s):")
        for i, ctx in duplicate_contexts:
            print(f"  - Duplicate at index {i}: {ctx[:75]}...")
    else:
        print("\nAll contexts are unique.")

    if duplicate_questions:
        print(f"\nFound {len(duplicate_questions)} duplicate question(s):")
        for i, q in duplicate_questions:
            print(f"  - Duplicate at index {i}: {q}")
    else:
        print("\nAll questions are unique.")

# Run the check
check_uniqueness("../../data/processed/easy_mcq_dataset.json")
