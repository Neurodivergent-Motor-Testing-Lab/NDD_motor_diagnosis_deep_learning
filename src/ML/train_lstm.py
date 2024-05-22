import scipy.io as sio
import numpy as np
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from ML.LSTM import LSTM
import torch
import matplotlib.pyplot as plt
from statistics import mean
from ML.move_dataset import MovementDataset

mean_test_accuracy_hist = []
test_X = None
test_y = None


def next_path(path_pattern):
    """
    Finds the next free path in an sequentially named list of folders


    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b


def organizeTrialData(data_dict, signal_idx):
    sequences = list()
    labels = list()
    for subject_diagnoses_str in data_dict.keys():
        for current_trial_data in data_dict[subject_diagnoses_str]:
            current_trial_data = current_trial_data[
                ~np.isnan(current_trial_data).any(axis=1)
            ]
            current_trial_data = current_trial_data[:, signal_idx]
            sequences.append(current_trial_data)
            labels.append(subject_diagnoses_str.replace("A^2", "ASD+ADHD"))

    return sequences, labels


def pruneSequences(sequences, targets, min_sequence_length):
    len_sequences = []
    unskipped_sequences = []
    unskipped_targets = []

    for one_seq, one_target in zip(sequences, targets):
        if len(one_seq) <= min_sequence_length:
            continue
        unskipped_sequences.append(one_seq)
        unskipped_targets.append(one_target)
        len_sequences.append(len(one_seq))
    min_length = min(len_sequences)
    print("Min Trial Length", min_length)
    print("Max Trial Length", max(len_sequences))
    print("Average Trial Length", mean(len_sequences))
    for i in range(len(unskipped_sequences)):
        old_array = unskipped_sequences[i]
        new_array = old_array[:min_length, :]
        unskipped_sequences[i] = new_array
    print("Num targets", len(unskipped_targets))
    return unskipped_sequences, unskipped_targets

    # for class in label_mappings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM.")
    input_signals = [
        "linear_accel",
        "rpy_dot",
        "rpy",
        "linear_accel_and_rpy_dot",
        "linear_accel_and_rpy",
        "rpy_dot_and_rpy",
        "all",
    ]
    parser.add_argument(
        "input_signal",
        type=str,
        default="rpy_dot_and_rpy",
        choices=input_signals,
        help="Select what input signal to use. Choices include "
        + ", ".join(input_signals),
    )
    parser.add_argument("--ip_file_path", help="Path to data input")
    parser.add_argument("--op_data_path", help="Path to data output")
    parser.add_argument("--num_epochs", type=int, default=750)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--num_lstm_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--min_sequence_length", type=int, default=105)
    parser.add_argument("--dropout", type=float, default=0.3)

    args = parser.parse_args()
    print(args)
    ip_file_path = "../data/xsens/data/all_trial_data.npy"
    if args.ip_file_path != None:
        ip_file_path = args.ip_file_path
    op_data_path = "../data/xsens/models/current_experiments/"
    if args.op_data_path != None:
        op_data_path = args.op_data_path

    if args.input_signal == "linear_accel":
        sigal_idx = [0, 1, 2]
    if args.input_signal == "rpy_dot":
        sigal_idx = [3, 4, 5]
    if args.input_signal == "rpy":
        sigal_idx = [6, 7, 8]
    if args.input_signal == "linear_accel_and_rpy_dot":
        sigal_idx = [0, 1, 2, 3, 4, 5]
    if args.input_signal == "linear_accel_and_rpy":
        sigal_idx = [0, 1, 2, 6, 7, 8]
    if args.input_signal == "rpy_dot_and_rpy":
        sigal_idx = [3, 4, 5, 6, 7, 8]
    if args.input_signal == "all":
        sigal_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    path_pattern = os.path.join(op_data_path, "Experiment-")
    path_pattern += "%s"
    op_experiment_path = next_path(path_pattern)
    if not os.path.exists(op_experiment_path):
        os.makedirs(op_experiment_path)
    with open(os.path.join(op_experiment_path, "settings.txt"), "w") as f:
        f.write(args.__str__())
    all_data = np.load(ip_file_path, allow_pickle=True)
    all_data = all_data.item()
    sequences, targets = organizeTrialData(all_data, sigal_idx)
    new_seq, targets = pruneSequences(sequences, targets, args.min_sequence_length)
    print("Num seq", len(new_seq))
    print("Num targets", len(targets))
    final_seq = np.stack(new_seq)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Uncomment this to run on GPU'
        print("DEVICE IS CUDA")
    else:
        device = torch.device("cpu")
        print("DEVICE IS CPU :()")

    num_features = final_seq.shape[2]
    dataset = MovementDataset(device, final_seq, targets)

    path = os.path.join(op_experiment_path, "dataset.pt")
    torch.save(dataset, path)

    num_outputs = np.prod(np.unique(dataset.y).shape)
    print(num_outputs)

    lstm = LSTM(
        args.num_folds,
        input_dim=num_features,
        output_dim=num_outputs,
        save_path=op_experiment_path,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_lstm_layers=args.num_lstm_layers,
        batch_size=args.batch_size,
        dropout=args.dropout,
    )
    lstm.train(dataset, num_epochs=args.num_epochs)
