from DL.train_lstm import runExperiment
import argparse
import os
import shutil
import time


def runAllForP(
    input_signal,
    num_experiments,
    config_output_folder,
    ip_file_path,
    op_data_path,
    test_ratio,
    min_sequence_length,
    desired_sequence_length,
    label_shuffle_probability,
    enable_early_stopping,
    early_stopping_patience,
    early_stopping_delta,
    num_folds,
    hidden_dims,
    learning_rates,
    weight_decays,
    num_lstm_layers,
    batch_size,
    dropout,
    num_epochs,
):
    for n in range(num_experiments):
        destination_path = os.path.join(
            config_output_folder, str(label_shuffle_probability), str(n), input_signal
        )
        if os.path.exists(destination_path) == False:
            os.makedirs(destination_path)
        training_output_folder = runExperiment(
            input_signal,
            ip_file_path,
            op_data_path,
            test_ratio,
            min_sequence_length,
            desired_sequence_length,
            label_shuffle_probability,
            enable_early_stopping,
            early_stopping_patience,
            early_stopping_delta,
            num_folds,
            hidden_dims,
            learning_rates,
            weight_decays,
            num_lstm_layers,
            batch_size,
            dropout,
            num_epochs,
        )
        for file_name in os.listdir(training_output_folder):
            shutil.move(
                os.path.join(training_output_folder, file_name),
                os.path.join(destination_path, file_name),
            )
        time.sleep(1)

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LSTM with random permutations of a subset of data."
    )
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
        default="all",
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
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[10, 20])
    parser.add_argument(
        "--learning_rates", nargs="+", type=float, default=[0.001, 0.0025, 0.005]
    )
    parser.add_argument(
        "--weight_decays", nargs="+", type=float, default=[0.001, 0.005, 0.01, 0.05]
    )
    parser.add_argument("--num_lstm_layers", nargs="+", type=int, default=[1, 2])
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
    parser.add_argument("--num_experiments", type=int, default=5)
    parser.add_argument(
        "--config_output_folder",
        type=str,
        default="../data/xsens/models/label_swapping/",
    )

    args = parser.parse_args()
    ip_file_path = "../data/xsens/data/all_trial_data.npy"
    if args.ip_file_path != None:
        ip_file_path = args.ip_file_path
    op_data_path = "../data/xsens/models/current_experiments/"
    if args.op_data_path != None:
        op_data_path = args.op_data_path
    label_shuffle_probabilities = [0.0, 0.25, 0.5, 0.75, 1.0]
    for label_shuffle_probability in label_shuffle_probabilities:
        runAllForP(
            args.input_signal,
            args.num_experiments,
            args.config_output_folder,
            ip_file_path,
            op_data_path,
            args.test_ratio,
            args.min_sequence_length,
            args.desired_sequence_length,
            label_shuffle_probability,
            args.enable_early_stopping,
            args.early_stopping_patience,
            args.early_stopping_delta,
            args.num_folds,
            args.hidden_dims,
            args.learning_rates,
            args.weight_decays,
            args.num_lstm_layers,
            args.batch_size,
            args.dropout,
            args.num_epochs,
        )
