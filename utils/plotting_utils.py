import os
import matplotlib.pyplot as plt
import numpy as np

from utils import post_processing_utils

def plot_confusion_matrix(file_path):
    matrices = post_processing_utils.compute_confusion_matrix(file_path)
    
    for idx, confusion_matrix in enumerate(matrices):
        matrix = np.array([
            [confusion_matrix["TP"], confusion_matrix["FN"]],
            [confusion_matrix["FP"], confusion_matrix["TN"]]
        ])

        fig, ax = plt.subplots()
        cax = ax.matshow(matrix, cmap=plt.cm.Blues)
        plt.title('Scene {}'.format(idx))
        fig.colorbar(cax)

        ax.set_xticklabels([''] + ['Predicted Positive', 'Predicted Negative'])
        ax.set_yticklabels([''] + ['Actual Positive', 'Actual Negative'])

        for (i, j), val in np.ndenumerate(matrix):
            ax.text(j, i, f'{val}', ha='center', va='center', color='red')
        
        if not os.path.exists("analysis"):
            os.makedirs("analysis")

        output_path = os.path.join("analysis", f"cm_scene_{idx}.png")
        plt.savefig(output_path)
        plt.close()

def plot_histograms(file_path):
    confusion_matrices = post_processing_utils.compute_confusion_matrix(file_path)
    
    tps = [matrix["TP"] for matrix in confusion_matrices]
    fps = [matrix["FP"] for matrix in confusion_matrices]
    tns = [matrix["TN"] for matrix in confusion_matrices]
    fns = [matrix["FN"] for matrix in confusion_matrices]

    if not os.path.exists("analysis"):
        os.makedirs("analysis")

    def plot_and_save_histogram(data, title, xlabel, filename):
        plt.figure()
        plt.hist(data, bins=20, color='blue', alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        output_path = os.path.join("analysis", filename)
        plt.savefig(output_path)
        plt.close()

    plot_and_save_histogram(tps, 'True Positives Histogram', 'True Positives', 'histogram_true_positives.png')
    plot_and_save_histogram(fps, 'False Positives Histogram', 'False Positives', 'histogram_false_positives.png')
    plot_and_save_histogram(tns, 'True Negatives Histogram', 'True Negatives', 'histogram_true_negatives.png')
    plot_and_save_histogram(fns, 'False Negatives Histogram', 'False Negatives', 'histogram_false_negatives.png')