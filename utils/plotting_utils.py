import os
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from utils import post_processing_utils
from pathlib import Path

def save_confusion_matrix_figure(matrix, plot_title, file_path, std_matrix = None):
    _, ax = plt.subplots()
    cmap = colors.ListedColormap(['White'])
    _ = ax.matshow(matrix, cmap=cmap)
    plt.title(plot_title)

    ax.set_xticklabels([''] + ['Predicted Positive', 'Predicted Negative'])
    ax.set_yticklabels([''] + ['Actual Positive', 'Actual Negative'])

    if(std_matrix is None):
        for (i, j), val in np.ndenumerate(matrix):
            ax.text(j, i, "M: "+ format(val, '.2f'), ha='center', va='center', color='red')
    else:
        #TODO: This part is buggy
        for (i, j), val, std_val in zip(np.ndenumerate(matrix), np.ndenumerate(std_matrix)):
            ax.text(j, i, "M: "+ format(val, '.2f') + "SD: " + format(std_val), ha='center', va='center', color='red')
      
    plt.savefig(file_path)
    plt.close()
    
def convert_confusion_dict_to_matrix(confusion_matrix):
    
    total_positives = confusion_matrix["TP"] + confusion_matrix["FN"]
    total_negatives = confusion_matrix["FP"] + confusion_matrix["TN"]
    
    matrix = np.array([
            [(confusion_matrix["TP"] / total_positives)*100, (confusion_matrix["FN"] / total_positives)*100],
            [(confusion_matrix["FP"] / total_negatives)*100, (confusion_matrix["TN"] / total_negatives)*100]
        ])
    return matrix

def plot_avg_confusion_matrix(results_dir_path, num_scenes=30):
    results_per_seed_paths = os.listdir(results_dir_path)
    
    matrices_per_seed = []
    
    for seed_path in results_per_seed_paths:
        matrix_per_scene = post_processing_utils.compute_confusion_matrix(results_dir_path + "/" + seed_path)
        matrices_per_seed.append(matrix_per_scene)
    
    # compute average matrix per seed
    avg_mat_per_scene = []
    for scene_num in range(num_scenes):
        avg_mat_per_scene.append(np.mean(np.array([convert_confusion_dict_to_matrix(matrices_per_seed[seed_idx][scene_num]) 
                                                   for seed_idx in range(len(results_per_seed_paths))]), axis=0))
    
    
    out_dir_path = Path("analysis").joinpath(Path(results_dir_path).name)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    for idx, matrix in enumerate(avg_mat_per_scene):
        plot_title = 'Average Confusion Matrix Across Seeds. Scene {}.'.format(scene_num)
        file_path = out_dir_path.joinpath(f"llm_seeds_avg_cm_scene_{idx}.png")
        save_confusion_matrix_figure(matrix=matrix, plot_title=plot_title, file_path=file_path)
    
    plot_title = 'Averaged Confusion Matrix Across Seeds Then Scenes'
    file_path = out_dir_path.joinpath(f"scenes_avg_llm_seeds_avg_cm.png")
    avg_final_matrix = np.mean(np.array(avg_mat_per_scene), axis=0)
    std_final_matrix = np.std(np.array(avg_mat_per_scene), axis=0)
    
    save_confusion_matrix_figure(matrix=avg_final_matrix, plot_title=plot_title, file_path=file_path)


# # TODO: we want variation across llm seeds per scene not the other way around
# def plot_histograms(file_path, exp_name):
#     confusion_matrices = post_processing_utils.compute_confusion_matrix(file_path)
    
#     tps = [matrix["TP"] for matrix in confusion_matrices]
#     fps = [matrix["FP"] for matrix in confusion_matrices]
#     tns = [matrix["TN"] for matrix in confusion_matrices]
#     fns = [matrix["FN"] for matrix in confusion_matrices]

#     if not os.path.exists("analysis"):
#         os.makedirs("analysis")

#     def plot_and_save_histogram(data, title, xlabel, filename):
#         plt.figure()
#         plt.hist(data, bins=20, color='blue', alpha=0.7)
#         plt.title(title)
#         plt.xlabel(xlabel)
#         plt.ylabel('Frequency')
#         output_path = os.path.join("analysis", filename)
#         plt.savefig(output_path)
#         plt.close()

#     plot_and_save_histogram(tps, 'True Positives Histogram', 'True Positives', 'histogram_true_positives.png')
#     plot_and_save_histogram(fps, 'False Positives Histogram', 'False Positives', 'histogram_false_positives.png')
#     plot_and_save_histogram(tns, 'True Negatives Histogram', 'True Negatives', 'histogram_true_negatives.png')
#     plot_and_save_histogram(fns, 'False Negatives Histogram', 'False Negatives', 'histogram_false_negatives.png')
