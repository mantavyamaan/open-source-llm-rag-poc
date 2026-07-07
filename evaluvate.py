import json
from rag import ask_base_model, ask_rag_model


EVAL_FILE = "eval/questions.json"


def contains_expected_answer(answer: str, expected: str) -> bool:
    answer_lower = answer.lower()
    expected_keywords = expected.lower().split()

    matched_keywords = 0

    for word in expected_keywords:
        if len(word) > 3 and word in answer_lower:
            matched_keywords += 1

    return matched_keywords >= max(2, len(expected_keywords) // 3)


def evaluate():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    base_correct = 0
    rag_correct = 0

    results = []

    for item in questions:
        question = item["question"]
        expected = item["expected_answer"]

        print(f"\nQuestion: {question}")

        base_answer = ask_base_model(question)
        rag_result = ask_rag_model(question)
        rag_answer = rag_result["answer"]

        base_is_correct = contains_expected_answer(base_answer, expected)
        rag_is_correct = contains_expected_answer(rag_answer, expected)

        if base_is_correct:
            base_correct += 1

        if rag_is_correct:
            rag_correct += 1

        results.append({
            "question": question,
            "expected": expected,
            "base_answer": base_answer,
            "rag_answer": rag_answer,
            "base_correct": base_is_correct,
            "rag_correct": rag_is_correct,
            "sources": rag_result["sources"]
        })

        print(f"Expected: {expected}")
        print(f"Base Correct: {base_is_correct}")
        print(f"RAG Correct: {rag_is_correct}")

    total = len(questions)

    print("\nEvaluation Summary")
    print("------------------")
    print(f"Total Questions: {total}")
    print(f"Base Model Correct: {base_correct}/{total}")
    print(f"RAG Model Correct: {rag_correct}/{total}")
    print(f"Base Accuracy: {base_correct / total * 100:.2f}%")
    print(f"RAG Accuracy: {rag_correct / total * 100:.2f}%")

    with open("eval/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    evaluate()
