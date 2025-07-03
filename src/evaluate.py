import evaluate
import numpy as np

def compute_bleu(predictions, references):
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=[p.split() for p in predictions],
                       references=[[r.split()] for r in references])

def simulate_user_study(model, test_questions):
    """Simulate user queries and measure accuracy."""
    correct = 0
    for q in test_questions:
        answer = model.generate_answer(q)[0]
        # Dummy validation (replace with actual checks)
        if "def" in answer or "import" in answer:
            correct += 1
    return correct / len(test_questions)
