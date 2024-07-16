from llm_prediction_experiment import LLMPredictionExperiment
import argparse
import json
from pathlib import Path
import os
EXPERIMENT_OUT_DIR = "experiment_outputs"

'''
This script gets predictions from a specified LLM for a specified task and LLM seed.
A task here refers to a themed collection of many questions for many scenes that are stored in a database.
'''

def main(model_path, task_db_path, exp_name, seed):

    exp = LLMPredictionExperiment(task_db_path=task_db_path,
                                  model_path=model_path,
                                  llm_seed=seed)
    
    
    print("Start predictions >>>>")
    pred_qa = exp.predict()
    results_path = EXPERIMENT_OUT_DIR+"/"+exp_name+"_results.json"
    Path(EXPERIMENT_OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        res = {"model_path": model_path, "task_db_path": task_db_path, "seed": seed,
               "inputs_and_preds": pred_qa}
        json.dump(res, f)


if __name__ in "__main__":
    # Example params:
    # task_db_path = "datasets/kinematics_db"
    # model_path = "/network/weights/llama.var/llama_3/Meta-Llama-3-8B-Instruct"
    # exp_name = "kinematics_experiment_v0"
    # seed = 0
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('task_db_path')
    parser.add_argument('model_path')
    parser.add_argument('exp_name')
    parser.add_argument('seed', type=int)
    
    args = parser.parse_args()

    main(model_path=args.model_path, task_db_path=args.task_db_path,
         exp_name=args.exp_name, seed=args.seed)