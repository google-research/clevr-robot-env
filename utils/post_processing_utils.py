import json

def compute_confusion_matrix(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    confusion_matrices = []

    for item in data["inputs_and_preds"]:
        answers = item["answers"]
        gt = item["gt"]

        true_positives = sum(1 for a, g in zip(answers, gt) if a == g == True)
        false_positives = sum(1 for a, g in zip(answers, gt) if a == True and g == False)
        true_negatives = sum(1 for a, g in zip(answers, gt) if a == g == False)
        false_negatives = sum(1 for a, g in zip(answers, gt) if a == False and g == True)

        confusion_matrix = {
            "TP": true_positives,
            "FP": false_positives,
            "TN": true_negatives,
            "FN": false_negatives
        }

        confusion_matrices.append(confusion_matrix)

    return confusion_matrices
