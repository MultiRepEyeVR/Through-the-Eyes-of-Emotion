import os
import json
import torch
import re
from multirepresentation_model.model import MultiRepresentationModel


def load_create_model(config, fold_idx, num_classes):
    model_path = config['model_path']
    if model_path is not None:
        model_path = os.path.join(model_path, f'fold_{fold_idx}')
        with open(os.path.join(model_path, 'vivit_settings.json'), 'r') as file:
            vit_params = json.load(file)
        with open(os.path.join(model_path, 'ts_transformer_settings.json'), 'r') as file:
            ts_transformer_params = json.load(file)

        train_params = {
            "batch_size": config['batch_size'],
            "max_update_epochs": config['max_update_epochs'],
            "warmup_epochs": config['warmup_epochs'],
            "base_lr": config['base_lr'],
            "final_lr": config['final_lr'],
            "warmup_begin_lr": config['warmup_begin_lr'],
            "beta_1": config['beta_1'],
            "beta_2": config['beta_2'],
            "label_smoothing": config['label_smoothing'],
        }

        model = MultiRepresentationModel(vit_config=vit_params, ts_transformer_config=ts_transformer_params)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model_name = determine_model_to_load(model_path)
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(os.path.join(model_path, model_name), map_location=device, weights_only=False),)
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")
        print(f"{model_name} loaded successfully from pretrained folder!")
    else:
        # Model parameters
        vit_params = {
            "image_size": config['frame_size'],
            "image_patch_size": config['vivit']['image_patch_size'],
            "frames": int(config['fps'] * config['window_time']),
            "frame_patch_size": config['vivit']['frame_patch_size'],
            "dim": config['vivit']['dim'],
            "spatial_depth": config['vivit']['spatial_depth'],
            "temporal_depth": config['vivit']['temporal_depth'],
            "heads": config['vivit']['heads'],
            "mlp_dim": config['vivit']['mlp_dim'],
            "pool": config['vivit']['pooling'],
            "channels": config['frame_channel'],
            "dim_head": config['vivit']['dim_head'],
            "enc_dropout": config['vivit']['enc_dropout_rate'],
            "emb_dropout": config['vivit']['emb_dropout_rate'],
            "variant": config['vivit']['variant'],
        }

        if config['ts_transformer']['model']['conv_config']:
            conv_config = config['ts_transformer'][config['ts_transformer']['model']['conv_config']]
        else:
            conv_config = None

        ts_transformer_params = {
            "feat_dim": config['ts_transformer']['model']['feat_dim'],
            "max_len": int(config["original_fps"] * config["window_time"]),
            "d_model": config['ts_transformer']['model']['d_model'],
            "n_heads": config['ts_transformer']['model']['n_heads'],
            "num_layers": config['ts_transformer']['model']['n_layers'],
            "dim_feedforward": config['ts_transformer']['model']['d_ff'],
            "enc_dropout": config['ts_transformer']['model']['enc_dropout'],
            "emb_dropout": config['ts_transformer']['model']['emb_dropout'],
            "embedding": config['ts_transformer']['model']['projection'],
            "conv_config": conv_config,
            "output_pool": config['ts_transformer']['model']['output_pool'],
        }

        train_params = {
            "batch_size": config['batch_size'],
            "max_update_epochs": config['max_update_epochs'],
            "warmup_epochs": config['warmup_epochs'],
            "base_lr": config['base_lr'],
            "final_lr": config['final_lr'],
            "warmup_begin_lr": config['warmup_begin_lr'],
            "beta_1": config['beta_1'],
            "beta_2": config['beta_2'],
            "label_smoothing": config['label_smoothing'],
        }

        model = MultiRepresentationModel(vit_config=vit_params, ts_transformer_config=ts_transformer_params)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    return model, vit_params, ts_transformer_params, train_params, device


def determine_model_to_load(model_path):
    confusion_matrix_files = [file for file in os.listdir(model_path) if file.endswith('confusion_matrix.png')]

    f1_scores = {}
    for file in confusion_matrix_files:
        match = re.search(r'f1_(\d+\.\d+)', file)
        if match:
            f1_score = float(match.group(1))
            f1_scores[file] = f1_score

    if len(f1_scores) == 2:
        best_file = [file for file in f1_scores if 'best' in file][0]
        last_file = [file for file in f1_scores if 'last' in file][0]
        if f1_scores[last_file] >= f1_scores[best_file]:
            return 'last_model.pth'
        else:
            return 'best_model.pth'
    else:
        return 'last_model.pth'