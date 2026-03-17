import csv
import json
from statistics import mean

from funtions import (
    compute_rouge,
    compute_bleu,
    compute_bertscore,
    is_response_safe,
    is_medical_correctness
)
from my_model import Test_model


def evaluate_model_on_csv(model, csv_path):
    results = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader, start=1):
            user_input = row["input"].strip()
            reference_answer = row["output"].strip()

            try:
                tested_answer = model.generate(user_input)
                if tested_answer is None:
                    tested_answer = ""
                tested_answer = str(tested_answer).strip()
            except Exception as e:
                print(f"[Row {idx}] generate() failed: {e}")
                tested_answer = ""

            # ROUGE
            try:
                rouge_l = compute_rouge(tested_answer, reference_answer)
            except Exception as e:
                print(f"[Row {idx}] ROUGE failed: {e}")
                rouge_l = 0.0

            # BLEU
            try:
                bleu = compute_bleu(tested_answer, reference_answer)
            except Exception as e:
                print(f"[Row {idx}] BLEU failed: {e}")
                bleu = 0.0

            # BERTScore
            try:
                bertscore = compute_bertscore(tested_answer, reference_answer)
            except Exception as e:
                print(f"[Row {idx}] BERTScore failed: {e}")
                bertscore = 0.0

            # Safety check
            try:
                is_safe = is_response_safe(
                    user_input,
                    reference_answer,
                    tested_answer
                )
            except Exception as e:
                print(f"[Row {idx}] Safety judge failed: {e}")
                is_safe = False

            # Medical correctness check
            try:
                is_correct = is_medical_correctness(
                    user_input,
                    reference_answer,
                    tested_answer
                )
            except Exception as e:
                print(f"[Row {idx}] Correctness judge failed: {e}")
                is_correct = False

            row_result = {
                "row_id": idx,
                "input": user_input,
                "reference_answer": reference_answer,
                "tested_answer": tested_answer,
                "ROUGE-L": rouge_l,
                "BLEU": bleu,
                "BERTScore": bertscore,
                "SAFE": is_safe,
                "MEDICAL_CORRECT": is_correct,
            }

            results.append(row_result)

            print(f"[Row {idx}] done | "
                  f"ROUGE-L={rouge_l:.4f}, "
                  f"BLEU={bleu:.4f}, "
                  f"BERTScore={bertscore:.4f}, "
                  f"SAFE={is_safe}, "
                  f"MEDICAL_CORRECT={is_correct}")

    return results


def summarize_results(results):
    if not results:
        return {
            "num_samples": 0,
            "avg_rouge_l": 0.0,
            "avg_bleu": 0.0,
            "avg_bertscore": 0.0,
            "safety_pass_rate": 0.0,
            "medical_correct_rate": 0.0,
        }

    avg_rouge_l = mean(item["ROUGE-L"] for item in results)
    avg_bleu = mean(item["BLEU"] for item in results)
    avg_bertscore = mean(item["BERTScore"] for item in results)

    safety_pass_rate = mean(1 if item["SAFE"] else 0 for item in results)
    medical_correct_rate = mean(1 if item["MEDICAL_CORRECT"] else 0 for item in results)

    return {
        "num_samples": len(results),
        "avg_rouge_l": avg_rouge_l,
        "avg_bleu": avg_bleu,
        "avg_bertscore": avg_bertscore,
        "safety_pass_rate": safety_pass_rate,
        "medical_correct_rate": medical_correct_rate,
    }


def save_results_to_json(results, summary, output_path="evaluation_results.json"):
    data = {
        "summary": summary,
        "details": results
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    csv_path = "sample_30_rows.csv"

    model = Test_model()

    results = evaluate_model_on_csv(model, csv_path)
    summary = summarize_results(results)

    print("\n===== Final Summary =====")
    print(f"Number of samples       : {summary['num_samples']}")
    print(f"Average ROUGE-L         : {summary['avg_rouge_l']:.4f}")
    print(f"Average BLEU            : {summary['avg_bleu']:.4f}")
    print(f"Average BERTScore       : {summary['avg_bertscore']:.4f}")
    print(f"Safety pass rate        : {summary['safety_pass_rate']:.4%}")
    print(f"Medical correct rate    : {summary['medical_correct_rate']:.4%}")

    save_results_to_json(results, summary)


if __name__ == "__main__":
    main()