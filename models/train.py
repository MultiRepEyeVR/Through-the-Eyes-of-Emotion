import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import gc
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def train_model(model, loss, optimizer, scheduler, loaded_data, vivit_config, ts_config, train_config, config, fold_idx):
    train_set = loaded_data[0]
    val_set = loaded_data[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=============================================================\n"
          f"=====================Training via {device}===================\n"
          f"=============================================================")

    path = os.path.join(config['result_path'], f"baseline_{config['training_proportion']}_train")

    path = os.path.join(path, f"{config['cross_validation_fold']}_fold_test")

    path = os.path.join(path, f"window_{config['window_time']}_periocular_rate_{config['periocular_rate']}_frame_size_{config['frame_size']}_"
                              f"image_patch_size_{vivit_config['image_patch_size']}_frame_patch_size_{vivit_config['frame_patch_size']}_"
                              f"overlap_{config['overlap']}")

    path = os.path.join(path, f"epochs_{config['epochs']}_max_update_epochs_{config['max_update_epochs']}_warmup_epochs_{config['warmup_epochs']}_"
                              f"batch_size_{config['batch_size']}_"
                              f"base_lr_{format(config['base_lr'], '.10f').rstrip('0').rstrip('.')}_"
                              f"warmup_begin_lr_{format(config['warmup_begin_lr'], '.10f').rstrip('0').rstrip('.')}_"
                              f"final_lr_{format(config['final_lr'], '.10f').rstrip('0').rstrip('.')}_"
                              f"label_smoothing_{format(config['label_smoothing'], '.10f').rstrip('0').rstrip('.')}")

    path = os.path.join(path, f"fold_{fold_idx+1}")

    config["model_path"] = path

    log_dir = os.path.join(path, f"TensorBoard_Log")
    writer = SummaryWriter(log_dir)

    weight_log_dir = os.path.join(path, f"Weight_Log")
    os.makedirs(weight_log_dir, exist_ok=True)

    print(f'Run cmd: tensorboard --logdir={log_dir} then open http://localhost:6006')

    with open(os.path.join(path, 'vivit_settings.json'), 'w') as f:
        json.dump(vivit_config, f, indent=4)

    with open(os.path.join(path, 'ts_transformer_settings.json'), 'w') as f:
        json.dump(ts_config, f, indent=4)

    with open(os.path.join(path, 'train_settings.json'), 'w') as f:
        json.dump(train_config, f, indent=4)

    model = model.to(device)

    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    val_f1_score_list = []

    # best_val_acc = 0  # Initialize variable to keep track of the best validation accuracy
    best_val_f1 = 0

    for epoch in range(1, config["epochs"] + 1):
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        train_loss, val_loss, val_acc, val_f1 = pass_epoch(model, loss, optimizer, scheduler, train_set, val_set, device)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_acc)
        val_f1_score_list.append(val_f1)

        # Log training and validation metrics
        writer.add_scalar('Train_Loss/train', train_loss, epoch)
        writer.add_scalar('Val_Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1_Score/val', val_f1, epoch)

        # Save the best model based on f1 accuracy
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(), os.path.join(config["model_path"], "best_model.pth")
            )

        model.cross_attention.log_weights(epoch, os.path.join(weight_log_dir, 'modality_and_pooling_weights.txt'))

    # Save the model for continuing the training
    torch.save(
        model.state_dict(), os.path.join(config["model_path"], "last_model.pth")
    )

    save_metrics(train_loss_list, val_loss_list, val_accuracy_list, val_f1_score_list, config)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return writer


def eval_model(model, test_set, config, label_map, writer, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(os.path.join(config["model_path"], f"{model_name}_model.pth"), weights_only=False))
    model = model.to(device)
    model.eval()

    true_labels = []
    pred_labels = []

    for i, batch in enumerate(test_set):
        inputs_mvts, inputs_eye0, inputs_eye1, labels = batch

        inputs_mvts, inputs_eye0, inputs_eye1, labels = inputs_mvts.to(device), inputs_eye0.to(device), inputs_eye1.to(device), labels.to(device)

        predictions = model(inputs_eye0, inputs_eye1, inputs_mvts)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(torch.argmax(predictions, dim=1).cpu().numpy())

    test_acc = accuracy_score(true_labels, pred_labels)
    test_f1 = f1_score(true_labels, pred_labels, average='weighted')  # Use 'weighted' if you have imbalanced classes

    if writer:
        writer.add_scalar(f'Test/{model_name}_Accuracy', test_acc, global_step=None)
        writer.add_scalar(f'Test/{model_name}_F1_Score', test_f1, global_step=None)

    true_labels_decoded = [label_map[label] for label in true_labels]
    pred_labels_decoded = [label_map[label] for label in pred_labels]

    cm = confusion_matrix(true_labels_decoded, pred_labels_decoded, labels=list(label_map.values()))

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=label_map.values(), yticklabels=label_map.values())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    plt.savefig(
        os.path.join(config["model_path"], f"f1_{format(test_f1, '.5f').rstrip('0').rstrip('.')}_"
                                           f"acc_{format(test_acc, '.5f').rstrip('0').rstrip('.')}_"
                                           f"{model_name}_confusion_matrix.png"))

    if writer:
        writer.add_figure(f'Test/{model_name}_Confusion_Matrix', fig, global_step=None)

        writer.flush()
        writer.close()

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return test_acc, test_f1


def pass_epoch(model, loss, optimizer, scheduler, train_set, val_set, device):
    # Train the model
    model = model.train()
    train_loss = 0
    total_train_samples = 0

    for i, batch in enumerate(train_set):
        inputs_mvts, inputs_eye0, inputs_eye1, labels = batch
        inputs_mvts, inputs_eye0, inputs_eye1, labels = inputs_mvts.to(device), inputs_eye0.to(device), inputs_eye1.to(device), labels.to(device)

        outputs = model(inputs_eye0, inputs_eye1, inputs_mvts)
        compute_loss = loss(outputs, labels)

        batch_loss = torch.sum(compute_loss)
        mean_loss = batch_loss / len(compute_loss)

        optimizer.zero_grad()

        mean_loss.backward()

        optimizer.step()

        with torch.no_grad():
            total_train_samples += len(compute_loss)
            train_loss += batch_loss.item()

    if scheduler:
        scheduler.step()

    train_loss /= total_train_samples

    # Evaluate the model
    model = model.eval()
    val_loss = 0
    total_val_samples = 0

    # Initialize lists to store true and predicted labels
    true_labels = []
    pred_labels = []

    for i, batch in enumerate(val_set):
        inputs_mvts, inputs_eye0, inputs_eye1, labels = batch

        inputs_mvts, inputs_eye0, inputs_eye1, labels = inputs_mvts.to(device), inputs_eye0.to(device), inputs_eye1.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs_eye0, inputs_eye1, inputs_mvts)
            compute_loss = loss(outputs, labels)

            batch_loss = torch.sum(compute_loss)

        # Collect true and predicted labels
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        total_val_samples += len(compute_loss)
        val_loss += batch_loss.item()

    val_loss /= total_val_samples

    val_acc = accuracy_score(true_labels, pred_labels)
    val_f1 = f1_score(true_labels, pred_labels, average='weighted')  # Use 'weighted' if imbalanced classes

    return train_loss, val_loss, val_acc, val_f1


def save_metrics(train_loss_list, val_loss_list, val_accuracy_list, val_f1_score_list, config):
    epochs = range(1, config["epochs"] + 1)

    plt.figure(figsize=(12, 4))
    plt.plot(epochs, train_loss_list, label='Train Loss', linestyle='-')
    plt.plot(epochs, val_loss_list, label='Validation Loss', linestyle='-')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config["model_path"], "loss_plot.png"))

    plt.clf()

    plt.figure(figsize=(12, 4))
    plt.plot(epochs, val_accuracy_list, label='Validation Accuracy', linestyle='-')
    plt.plot(epochs, val_f1_score_list, label='Validation F1 Score', linestyle='-')
    plt.title('Validation Accuracy and F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config["model_path"], "accuracy_f1_plot.png"))