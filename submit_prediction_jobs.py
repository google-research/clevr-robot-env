import argparse
import json
import math
import os
import time
from pathlib import Path


def create_script(job_spec_dict, path_to_code, model_path, task_db_path, exp_name, seeds_list):
    

        
        script_prelims = """#!/bin/bash
        #SBATCH --job-name={}
        #SBATCH --output={}
        #SBATCH --error={}
        #SBATCH --time={}
        #SBATCH --mem={}
        #SBATCH --gres=gpu:{}:{}""".format(exp_name, job_spec_dict["slurm_out_dir"],
                                      job_spec_dict["slurm_error_dir"],
                                      job_spec_dict["time"],
                                      job_spec_dict["mem"],
                                      job_spec_dict["gpu_type"],
                                      job_spec_dict["num_gpus"])
        script_env_prep = """cd {}
        module --quiet purge
        module load python/3.10
        export MUJOCO_GL="osmesa"
        virtualenv $SLURM_TMPDIR/clevrenv
        source $SLURM_TMPDIR/clevrenv/bin/activate
        pip install -r requirements.txt
        """.format(path_to_code)
        
        script_core = '\n'.join(["python run_llm_prediction_experiment.py {} {} {} {}".format(task_db_path,
                                                            model_path,
                                                            exp_name,
                                                            seed) for seed in seeds_list])
        
        script = script_prelims + "\n" + script_env_prep + "\n" + script_core
        return script

def distribute_jobs(cluster_config_path, path_to_code, model_path, task_db_path, exp_name, num_seeds):
    
    with open(cluster_config_path) as f:
        job_spec_dict = json.load(f)
    
    #TODO: this won't work ... need to parse time
    assert (job_spec_dict["time_per_exp"] < job_spec_dict["max_job_time"])
    # do only 80 percent of actual max workload for job to leave buffer time
    max_seeds_per_job = math.floor((job_spec_dict["max_job_time"] / job_spec_dict["time_per_exp"])*0.8) 
    print("Preparing to launch jobs with", max_seeds_per_job, "experiments per job.")

            
    for i in range(0, num_seeds, max_seeds_per_job):
        
        seeds_list = [s for s in range(i, i + max_seeds_per_job)]
        print("Launching jobs for seeds", seeds_list)
        script_name = "auto_slurm.sh"
        
        Path(os.path.dirname(job_spec_dict["slurm_out_dir"])).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(job_spec_dict["slurm_error_dir"])).mkdir(parents=True, exist_ok=True)
        script = create_script(job_spec_dict, path_to_code, model_path, task_db_path, exp_name, seeds_list)
        
        # with open(script_name) as f:
        #     f.write(script)
        print(script)
        print("-----------")
        
        # os.system("bash {}".format(script_name))
        # os.remove(script_name)
        # time.sleep(2) # trying to be kind to slurm scheduler

    
if __name__ in "__main__":
    # Example params:
    # task_db_path = "datasets/kinematics_db"
    # model_path = "/network/weights/llama.var/llama_3/Meta-Llama-3-8B-Instruct"
    # exp_name = "kinematics_experiment_v0"
    # cluster_config_path = "clusters/mila.json"
    # num_llm_seeds = 5
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('task_db_path')
    parser.add_argument('model_path')
    parser.add_argument('exp_name')
    parser.add_argument('cluster_config_path')
    parser.add_argument('num_llm_seeds', type=int)
    
    
    args = parser.parse_args()

    distribute_jobs(cluster_config_path=args.cluster_config_path, 
         model_path=args.model_path, task_db_path=args.task_db_path,
         exp_name=args.exp_name, num_llm_seeds=args.num_llm_seeds)