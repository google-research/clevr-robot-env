import pandas as pd
 
def to_csv(all_preds):
    df = pd.DataFrame(columns=['scene_number', 'question_index', 'predicted_answer', 'ground_truth_answer'])
    for scene_idx, pred in enumerate(all_preds):
        description = pred["description"]
        for q_idx, q in enumerate(pred["questions"]):
            df.loc[len(df.index)] = [scene_idx, q_idx, pred["answers"][q_idx], pred["ground_truths"][q_idx]]
    df.to_csv('predictions.csv', encoding='utf-8', index=False)