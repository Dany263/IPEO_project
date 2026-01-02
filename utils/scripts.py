import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def log_experiment(file_path, experiment_dict):
    """
    Logs the results of a single experiment to a CSV file.

    If the file exists, it appends the new experiment, overwriting any previous entry
    with the same model and hyperparameter combination. Missing keys in the existing
    CSV are filled with NaN to maintain consistency.

    Args:
        file_path (str): Path to the CSV log file.
        experiment_dict (dict): Dictionary containing experiment results and hyperparameters.
    """
    # Ensure all keys exist in the dict and fill missing with NaN
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=experiment_dict.keys())

    # Fill any missing keys with NaN to match existing columns
    for col in df.columns:
        if col not in experiment_dict:
            experiment_dict[col] = pd.NA

    # Append the new experiment (overwrite if same combination)
    mask = (
        (df['model_name'] == experiment_dict['model_name']) &
        (df['dropout_rate'] == experiment_dict['dropout_rate']) &
        (df['learning_rate'] == experiment_dict['learning_rate']) &
        (df['batch_size'] == experiment_dict['batch_size']) &
        (df['weight_decay'] == experiment_dict['weight_decay']) &
        (df['weighted_ce'] == experiment_dict['weighted_ce']) &
        (df['n_train_samples'] == experiment_dict['n_train_samples'])
    )
    df = df[~mask]  # remove previous entry if same combination
    df = pd.concat([df, pd.DataFrame([experiment_dict])], ignore_index=True)
    
    df.to_csv(file_path, index=False)
    

def base_model_name(name):
    """
    Extracts the base model name from a full experiment name.

    For combined models, it includes the image backbone (e.g., "Combined_Resnet50").
    For other models, it returns just the main model name (e.g., "TabularSkip").

    Args:
        name (str): Full model name string, possibly including hyperparameter index.

    Returns:
        str: Base model name suitable for grouping or plotting.
    """
    parts = name.split('_')
    if parts[0].lower() == "combined":
        return "Combined_" + parts[1]  # e.g., Combined_Resnet50
    else:
        return parts[0]  # e.g., TabularSkip


def select_best_experiments(log_path):
    """
    Selects the best hyperparameter combination per base model from the experiment log.
    
    Args:
        log_path (str): Path to the CSV file containing experiment results.
        
    Returns:
        pd.DataFrame: Best experiment per base model, sorted by model name.
    
    Description:
        - Strips the hyperparameter index from model names (and includes image backbone for combined models).
        - Groups experiments by base model and selects the one with the highest validation Macro F1.
        - Return a table of the best experiments per model.
    """
    if not os.path.exists(log_path):
        print(f"No log file found at {log_path}")
        return None

    experiments_df = pd.read_csv(log_path)
    experiments_df['base_model'] = experiments_df['model_name'].apply(base_model_name)

    best_per_model = experiments_df.loc[
        experiments_df.groupby('base_model')['val_macro_f1'].idxmax()
    ].sort_values(by='model_name')

    return best_per_model


def plot_experiments(log_path, save_path=None):
    """
    Plots validation scores for all experiments in the log, showing hyperparameter indices.
    
    Args:
        log_path (str): Path to the CSV file containing experiment results.
        
    Description:
        - Groups experiments by base model on the x-axis.
        - Displays hyperparameter index numbers above Macro F1 scores (black) and below Accuracy scores (blue).
        - Automatically scales the y-axis using invisible points.
        - Includes a legend for Macro F1, Accuracy, and hyperparameter index numbers.
    """
    if not os.path.exists(log_path):
        print(f"No log file found at {log_path}")
        return

    experiments_df = pd.read_csv(log_path)
    experiments_df['base_model'] = experiments_df['model_name'].apply(base_model_name)

    plt.figure(figsize=(12,6))
    base_models = experiments_df['base_model'].unique()
    x_pos = {model: i for i, model in enumerate(base_models)}
    margin = 0.3
    plt.xlim(-margin, len(base_models)-1 + margin)

    # Invisible points for autoscaling y-axis
    for _, row in experiments_df.iterrows():
        x = x_pos[row['base_model']]
        plt.plot(x, row['val_macro_f1'], 'o', alpha=0)
        plt.plot(x, row['val_acc'], 'o', alpha=0)

    # Plot Macro F1 labels (top)
    for _, row in experiments_df.iterrows():
        x = x_pos[row['base_model']]
        plt.text(x, row['val_macro_f1'], row['model_name'].split('_')[-1], 
                 ha='center', va='bottom', color='black', fontsize=10)
    
    # Plot Accuracy labels (bottom)
    for _, row in experiments_df.iterrows():
        x = x_pos[row['base_model']]
        plt.text(x, row['val_acc'], row['model_name'].split('_')[-1], 
                 ha='center', va='top', color='blue', fontsize=10)

    # Legend
    black_patch = mpatches.Patch(color='black', label='Macro F1')
    blue_patch = mpatches.Patch(color='blue', label='Accuracy')
    dummy_point = plt.plot([], [], 'o', color='gray', label='Number = hyperparameter index')[0]
    plt.legend(handles=[black_patch, blue_patch, dummy_point], loc='best')

    plt.xticks(list(x_pos.values()), list(x_pos.keys()), rotation=45)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Validation Scores Across Experiments per Model")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()
