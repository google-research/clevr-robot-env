import transformers
import torch
from utils.db_utils import LLMDataset
from tqdm import tqdm
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

class LLMPredictionExperiment:
    def __init__(self, task_db_path, model_path, llm_seed=0) -> None:
        
        self.dataset = self._load_dataset(task_db_path)
        transformers.set_seed(llm_seed)
        self.pipeline = self._load_pipeline(model_path)
    
    def _load_dataset(self, task_db_path): 
        return LLMDataset(task_db_path)
    
    def _load_pipeline(self, model_path):
        return transformers.pipeline("text-generation", model=model_path, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    
    def predict(self): # TODO: make it possible to only do some preds, not all
        terminators = [self.pipeline.tokenizer.eos_token_id,
                       self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        
        all_preds = []
        scene_num = 0
        for scene_dict in iter(self.dataset):
            print("Scene num: ", scene_num)
            scene_answers = []
            for q in tqdm(scene_dict["questions"]):
                description = ' '.join(scene_dict['description'])
                messages = [[
                    {"role": "system", "content": f"You are looking at a table from the top. There are sphere objects on the table. {description}"},
                    {"role": "user", "content": f"{q} Answer only with 'True' or 'False' using reasoning from a 2d image of the scene and cardinal directions. The number of units isn't important."}
                ]]
                model_out = self.pipeline(messages, 
                                  max_new_tokens = 256,
                                  eos_token_id = terminators,
                                  do_sample = True,
                                  temperature = 0.6,
                                  top_p = 0.9)
                model_answer = model_out[0][0]["generated_text"][2]["content"]
                scene_answers.append(model_answer)
                
            all_preds.append({"description": description,
                              "questions": scene_dict["questions"],
                              "gt_answers": scene_dict["answers"],
                              "pred_answers": scene_answers})
            scene_num += 1
            
        return all_preds

    
