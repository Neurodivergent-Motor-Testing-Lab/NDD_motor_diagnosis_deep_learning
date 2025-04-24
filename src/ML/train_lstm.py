import numpy as np
import os
import argparse
from ML.LSTM_network import LSTMNetwork
import torch
from statistics import mean
from ML.move_datasets import MovementDataset
import random
from ML.early_stopping import EarlyStopping


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

# Remove trials that are too short
def pruneShortSequences(sequences, targets, min_sequence_length):
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

# Get a random sequence from each trial
def getRandomSequence(sequences, targets, desired_sequence_length):
    random_sequences = []
    for i in range(len(sequences)):
        old_array = sequences[i]
        initial_index = random.randint(0, len(old_array) - desired_sequence_length)
        new_array = old_array[
            initial_index : initial_index + desired_sequence_length, :
        ]
        random_sequences.append(new_array)
    return random_sequences, targets


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
    parser.add_argument(
        "--test_ratio", help="Amount of data for test set", type=float, default=0.2
    )
    parser.add_argument("--num_epochs", type=int, default=750)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--hidden_dims", nargs='+', type=int, default=[10, 20])
    parser.add_argument("--learning_rates", nargs='+', type=float, default=[0.001, 0.0025, 0.005])
    parser.add_argument("--weight_decays", nargs='+', type=float, default=[0.001, 0.005, 0.01, 0.05])
    parser.add_argument("--num_lstm_layers", nargs='+', type=int, default=[1, 2])
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--min_sequence_length", type=int, default=105)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--desired_sequence_length",
        help="Length of random sequence to use from trial",
        type=int,
        default=40,
    )
    parser.add_argument("--enable_early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_delta", type=float, default=0.005)
    parser.add_argument(
        "--label_shuffle_probability",
        help="Number of labels from each class to assign a random label",
        type=float,
        default=0.0,
    )
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
    new_seq, targets = pruneShortSequences(sequences, targets, args.min_sequence_length)
    new_seq, targets = getRandomSequence(new_seq, targets, args.desired_sequence_length)
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
    dataset = MovementDataset(
        device, final_seq, targets, args.label_shuffle_probability
    )

    path = os.path.join(op_experiment_path, "dataset.pt")
    torch.save(dataset, path)

    num_outputs = np.prod(np.unique(dataset.y).shape)
    print(num_outputs)

    lstm_network = LSTMNetwork(
        args.num_folds,
        input_dim=num_features,
        output_dim=num_outputs,
        save_path=op_experiment_path,
        hidden_dim_list=args.hidden_dims,
        learning_rate_list=args.learning_rates,
        weight_decay_list=args.weight_decays,
        num_lstm_layers_list=args.num_lstm_layers,
        batch_size=args.batch_size,
        dropout=args.dropout,
    )
    early_stopping_obj = None
    if args.enable_early_stopping:
        early_stopping_obj = EarlyStopping(
            patience=args.early_stopping_patience, delta=args.early_stopping_delta
        )
    lstm_network.train(
        dataset,
        num_epochs=args.num_epochs,
        test_ratio=args.test_ratio,
        early_stopping_obj=early_stopping_obj,
    )
