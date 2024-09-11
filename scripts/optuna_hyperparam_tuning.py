import optuna
import subprocess

def objective(trial):
    # Define the hyperparameters you want to tune
    num_epochs = trial.suggest_int("num_epochs", 10, 50)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    num_workers = trial.suggest_int("num_workers", 2, 8)
    limited_label = trial.suggest_uniform("limited_label", 0.1, 1.0)  # dataset usage ratio
    use_fp16 = trial.suggest_categorical("fp16", [True, False])

    # Create the command to run your training script with the current hyperparameters
    command = [
        'torchrun',
        '--nnodes=1',
        '--nproc_per_node=1',
        'run.py',
        '--config', 'configs/run/default.yaml',
        '--encoder_config', 'configs/foundation_models/prithvi.yaml',
        '--dataset_config', 'configs/datasets/mados.yaml',
        '--segmentor_config', 'configs/segmentors/upernet.yaml',
        '--augmentation_config', 'configs/augmentations/segmentation_default.yaml',
        '--num_workers', str(num_workers),
        '--eval_interval', '1',
        '--epochs', str(num_epochs),
        '--learning_rate', str(learning_rate),
        '--batch_size', str(batch_size),
        '--limited_label', str(limited_label),
        '--fp16' if use_fp16 else '',
        '--use_wandb'
    ]

    # Run the training process
    result = subprocess.run(command, capture_output=True, text=True)

    # Extract validation loss from the output
    validation_loss = extract_validation_loss(result.stdout)
    
    return validation_loss

def extract_validation_loss(output):
    for line in output.splitlines():
        if "Validation Loss" in line:
            return float(line.split()[-1])
    return None

# Create and optimize the study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Print the best trial details
print(f"Best trial: {study.best_trial.number}")
print(f"Best value (Validation Loss): {study.best_trial.value}")
print(f"Best hyperparameters: {study.best_trial.params}")

