import numpy as np
import torch
import argparse
from DL.LSTM_network import Net
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import shap


class ComputeSHAPValues:
    def __init__(
        self,
        model_path,
        training_set_path,
        test_set_path,
        output_path,
        feature_base_names,
        num_samples=100,
    ):
        shap.initjs()
        self.model_path = model_path
        self.training_set_path = training_set_path
        self.test_set_path = test_set_path
        self.output_path = os.path.join(output_path, "shap-samples" + str(num_samples))

        os.makedirs(self.output_path, exist_ok=True)
        # self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # Uncomment this to run on GPU'
        else:
            self.device = torch.device("cpu")

        self.feature_base_names = feature_base_names
        self.model = self.load_model()

        self.X_train, self.y_train = self.load_data(self.training_set_path)
        self.X_test, self.y_test = self.load_data(self.test_set_path)
        self.keras_model = self.convertToKeras()

        # self.compute_shap_values()
        explainer, shap_values = self.compute_keras_shap_values(num_samples)
        self.draw_all_plots(explainer, shap_values, num_samples)

    def convertToKeras(self):
        from interpret.keras_net import build_keras_net, transfer_weights

        keras_model = build_keras_net(
            self.model.input_dim,
            self.model.hidden_dim,
            self.model.output_dim,
            num_lstm_layers=self.model.num_lstm_layers,
            dropout=self.model.dropout_layer.p,
        )

        transfer_weights(
            self.model_path,
            keras_model,
            self.device,
            num_lstm_layers=self.model.num_lstm_layers,
        )

        return keras_model

    def wrapped_model(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32).permute(1, 0, 2).to("cuda")
        with torch.no_grad():
            output = self.model(x_tensor)
        return output.detach().cpu().numpy()

    def load_model(self):
        saved_data = torch.load(self.model_path, weights_only=False)
        model = Net(
            saved_data["input_dim"],
            saved_data["hidden_dim"],
            saved_data["output_dim"],
            saved_data["batch_size"],
            saved_data["num_lstm_layers"],
            saved_data["dropout"],
        )
        model.load_state_dict(saved_data["model_state_dict"])
        model.eval()
        model.to(self.device)
        print("Model loaded successfully.")
        return model

    def load_data(self, data_path):
        data_loader = torch.load(data_path, weights_only=False)

        x_size = data_loader.dataset.X[0].shape
        Xs = torch.empty(
            0, x_size[0], x_size[1], device=self.device, dtype=torch.float32
        )
        ys = torch.empty(0, device=self.device, dtype=torch.float32)
        for data in data_loader:
            Xs = torch.cat((Xs, data[0]), 0)
            ys = torch.cat((ys, data[1]), 0)
        print("Data loaded successfully.")

        Xs = Xs.transpose(1, 2)
        ys = ys.view(-1)
        return Xs.cpu().numpy(), ys.cpu().numpy()
        # return tf.convert_to_tensor(Xs.cpu().numpy()), tf.convert_to_tensor(ys.cpu().numpy())

    def compute_keras_shap_values(self, num_samples):

        explainer = shap.GradientExplainer(
            self.keras_model, self.X_train[:num_samples]
        )  # Use a subset of data to initialize explainer

        shap_values = explainer.shap_values(self.X_train[:num_samples])

        np.save(os.path.join(self.output_path, "shap_values.npy"), shap_values)
        return explainer, shap_values

    def draw_all_plots(self, explainer, shap_values, num_samples):
        class_names = ["ADHD", "ASD", "ASD+ADHD", "NT"]

        # Loop through each class
        for class_idx, class_name in enumerate(class_names):
            print(f"Processing class: {class_name}")

            # (samples, seq_len, input_dim)
            class_shap = shap_values[:, :, :, class_idx]  # shape: (n, 40, 9)
            
            # Average over timesteps (axis 1)
            shap_avg = np.mean(class_shap, axis=1)  # shape: (n_samples, 9)

            # Make the summary plot
            shap.summary_plot(shap_avg, self.X_train[:num_samples, 0, :], feature_names=self.feature_base_names, show=False)
            plt.title(f"SHAP Summary - Averaged Over Time - {class_name}")
            plt.savefig(os.path.join(self.output_path, f"shap_summary_avg_{class_name}.svg"), bbox_inches="tight")
            plt.close()


def plotForConfig(signal, experiment_path, num_samples):

    feature_base_names = [
        "linear_acc_x",
        "linear_acc_y",
        "linear_acc_z",
        "rpy_dot_x",
        "rpy_dot_y",
        "rpy_dot_z",
        "rpy_x",
        "rpy_y",
        "rpy_z",
    ]

    if signal == "linear_accel":
        sigal_idx = [0, 1, 2]
    if signal == "rpy_dot":
        sigal_idx = [3, 4, 5]
    if signal == "rpy":
        sigal_idx = [6, 7, 8]
    if signal == "linear_accel_and_rpy_dot":
        sigal_idx = [0, 1, 2, 3, 4, 5]
    if signal == "linear_accel_and_rpy":
        sigal_idx = [0, 1, 2, 6, 7, 8]
    if signal == "rpy_dot_and_rpy":
        sigal_idx = [3, 4, 5, 6, 7, 8]
    if signal == "all":
        sigal_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    features = [feature_base_names[i] for i in sigal_idx]
    data_path = os.path.join(experiment_path, "Final", "model-fold-0")

    network_path = os.path.join(data_path, "network.pt")
    training_set_path = os.path.join(data_path, "train_loader.pt")
    test_set_path = os.path.join(data_path, "evaluation_loader.pt")
    output_path = os.path.join(experiment_path, "figures", "Final", "shap_plots")
    ComputeSHAPValues(
        network_path,
        training_set_path,
        test_set_path,
        output_path,
        features,
        num_samples,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Shap values.")
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
    parser.add_argument(
        "--experiment_path",
        type=str,
        help="Path to the all experiments.",
        default="/home/khoshrav/Personal/Research/IU_ML/data/xsens/models/final_results",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to use for SHAP computation.",
    )
    args = parser.parse_args()
    plotForConfig(args.input_signal, args.experiment_path, args.num_samples)
