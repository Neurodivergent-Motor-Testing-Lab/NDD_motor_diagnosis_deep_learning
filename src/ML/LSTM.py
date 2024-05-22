import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
import matplotlib.pyplot as plt
import os
import math
from sklearn.utils import resample
import pickle
from .move_dataset import MovementDataset
from sklearn.metrics import roc_curve, auc


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()


# Here we define our self.net as a class
class Net(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        batch_size,
        num_lstm_layers=1,
        dropout=0.3,
    ):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_lstm_layers = num_lstm_layers
        # Define the LSTM layer
        self.lstm = nn.LSTM(
            self.input_dim, self.hidden_dim, self.num_lstm_layers, dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.normal_(self.linear.weight, mean=0, std=1.0)

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, _ = self.lstm(input)
        flattened_lstm_out = lstm_out[-1, :, :]
        dropout_op = self.dropout(flattened_lstm_out)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(dropout_op)
        # y_pred = F.softmax(y_pred, dim=0)
        return y_pred


class LSTM:
    def __init__(
        self,
        num_folds,
        input_dim=9,
        hidden_dim=20,
        output_dim=4,
        save_path="../data/models/",
        learning_rate=0.0005,
        weight_decay=0.005,
        num_lstm_layers=1,
        batch_size=50,
        dropout=0.3,
    ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # Uncomment this to run on GPU'
            print("DEVICE IS CUDA")
        else:
            self.device = torch.device("cpu")
            print("DEVICE IS CPU :()")

        self.num_folds = num_folds
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_lstm_layers = num_lstm_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.loss_function = nn.CrossEntropyLoss()

    def predict_probs(self, net, dataLoader):
        desired_labels, outputs, _ = self.predict(net, dataLoader)
        predictions = F.softmax(outputs, dim=0)
        return desired_labels, predictions

    def predict(self, net, dataLoader):
        prediction = None
        desired_labels = None
        total_loss = 0
        # Evaluationfor this fold
        net.eval()
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(dataLoader, 0):
                # Get inputs
                inputs, targets = data
                inputs = inputs.transpose(0, 2).transpose(1, 2)
                targets = targets.view(-1)

                if torch.isnan(inputs).any():
                    print("Input has nan! Quitting")
                    exit()
                # Generate outputs
                new_outputs = net(inputs)
                total_loss += self.loss_function(new_outputs, targets).item()
                if prediction is None:
                    prediction = new_outputs
                else:
                    prediction = torch.concat((prediction, new_outputs))

                if desired_labels is None:
                    desired_labels = targets
                else:
                    desired_labels = torch.concat((desired_labels, targets))
        return desired_labels, prediction, total_loss

    def plot_single_metric(self, ax, metric, evaluation_timestep):
        num_epochs = metric.shape[0]
        if num_epochs == 0:
            return
        x_range = np.arange(num_epochs) * evaluation_timestep
        ax.clear()
        for epoch_idx in range(metric.shape[1]):
            # print(epoch_idx)
            ax.plot(x_range, metric[:, epoch_idx], label=f"Fold {epoch_idx+1}")

        mean_metric = np.mean(metric, axis=1)
        ax.plot(x_range, mean_metric, label="Mean")
        ax.legend()
        # std_metric = np.std(metric, axis=1)
        # ax.fill_between(x_range, mean_metric-std_metric, mean_metric+std_metric, alpha=0.5)

    def plot_metrics(
        self,
        train_loss,
        test_loss,
        train_accuracy,
        test_accuracy,
        ax_train_loss,
        ax_test_loss,
        ax_train_accuracy,
        ax_test_accuracy,
        evaluation_timestep,
    ):
        self.plot_single_metric(ax_train_loss, train_loss, evaluation_timestep)
        ax_train_loss.set_title("Training Loss")
        ax_train_loss.set(xlabel="# of Epochs", ylabel="Loss")

        self.plot_single_metric(ax_test_loss, test_loss, evaluation_timestep)
        ax_test_loss.set_title("Test Loss")
        ax_test_loss.set(xlabel="# of Epochs", ylabel="Loss")

        self.plot_single_metric(ax_train_accuracy, train_accuracy, evaluation_timestep)
        ax_train_accuracy.set_ylim([0, 100.00])
        ax_train_accuracy.set_title("Training Accuracy")
        ax_train_accuracy.set(xlabel="# of Epochs", ylabel="Accuracy")
        ax_train_accuracy.set_aspect("auto")
        # ax_train_accuracy.set_ybound(lower=0.0, upper=1)

        # ax_test_accuracy.set_ylim([0, 100.00])
        self.plot_single_metric(ax_test_accuracy, test_accuracy, evaluation_timestep)
        ax_test_accuracy.set_title("Test Accuracy")
        ax_test_accuracy.set(xlabel="# of Epochs", ylabel="Accuracy")
        ax_test_accuracy.set_aspect("auto")
        # ax_test_accuracy.set_ybound(lower=0.0, upper=1)

        mng = plt.get_current_fig_manager()
        # mng.window.showMaximized()
        plt.pause(0.005)

    def evaluate(self, net, dataLoader):
        # Evaluationfor this fold
        correct, total = 0, 0
        net.eval()

        desired_labels, predictions, total_loss = self.predict(net, dataLoader)

        # Set total and correct
        _, predicted = torch.max(predictions, 1)
        total += desired_labels.size(0)
        correct += (predicted == desired_labels).sum().item()

        return total, correct, total_loss

    def train(self, dataset: MovementDataset, num_epochs=500, debug=True):
        evaluation_timestep = 10

        if debug:
            num_evaluations = math.floor(num_epochs / evaluation_timestep)
            train_accuracies = np.zeros((num_evaluations, self.num_folds))
            train_loss = np.zeros((num_evaluations, self.num_folds))
            test_accuracies = np.zeros((num_evaluations, self.num_folds))
            test_loss = np.zeros((num_evaluations, self.num_folds))

        final_train_loss = np.zeros(self.num_folds)
        final_test_loss = np.zeros(self.num_folds)
        final_train_accuracies = np.zeros(self.num_folds)
        final_test_accuracies = np.zeros(self.num_folds)
        if debug:
            all_tpr = {}
            num_roc_points = 500
            interp_fpr = np.linspace(0, 1, num_roc_points)

            all_auc = {}
            for ndd_class in dataset.diagnoses_mappings.keys():
                all_tpr[ndd_class] = np.zeros((self.num_folds, num_roc_points))
                all_auc[ndd_class] = np.zeros(self.num_folds)

            training_fig = plt.figure()
            ax_train_loss = training_fig.add_subplot(221)
            ax_train_loss.set_title("Training Loss", fontsize=15)

            ax_test_loss = training_fig.add_subplot(222, sharex=ax_train_loss)
            ax_test_loss.set_title("Test Loss", fontsize=15)

            ax_train_accuracy = training_fig.add_subplot(223, sharex=ax_train_loss)
            ax_train_accuracy.set_title("Training Accuracy", fontsize=15)
            ax_train_accuracy.set_ylim([0, 100])

            ax_test_accuracy = training_fig.add_subplot(
                224, sharex=ax_train_loss, sharey=ax_train_accuracy
            )
            ax_test_accuracy.set_title("Test Accuracy", fontsize=15)
            # ax_test_accuracy.set_ylim([0, 1])

            mean_roc_fig = plt.figure()

        kfold = StratifiedKFold(n_splits=self.num_folds, shuffle=True)
        splits = kfold.split(dataset, dataset.y)
        for fold, (train_ids, test_ids) in enumerate(splits):
            print(f"FOLD {fold}")
            print("--------------------------------")
            print("Num train", len(train_ids))
            print("Num Test", len(test_ids))
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, sampler=train_subsampler
            )
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, sampler=test_subsampler
            )
            if self.save_path is not None:
                fold_path = os.path.join(self.save_path, f"model-fold-{fold}")
                if not os.path.exists(fold_path):
                    os.makedirs(fold_path)

            net = Net(
                self.input_dim,
                self.hidden_dim,
                self.output_dim,
                self.batch_size,
                self.num_lstm_layers,
                self.dropout,
            )
            net.to(dataset.device)
            net.apply(reset_weights)
            # Initialize optimizer
            optimizer = torch.optim.Adam(
                net.parameters(), weight_decay=self.weight_decay, lr=self.learning_rate
            )
            # Run the training loop for defined number of epochs
            for epoch in range(0, num_epochs):

                # Set current loss value
                epoch_loss = 0
                # Iterate over the DataLoader for training data
                for i, data in enumerate(trainloader):

                    # Get inputs
                    inputs, targets = data
                    inputs = inputs.transpose(0, 2).transpose(1, 2)
                    targets = targets.view(-1)
                    net.train()
                    # Zero the gradients
                    optimizer.zero_grad()
                    # Initialise hidden state
                    # Don't do this if you want your LSTM to be stateful
                    # Perform forward pass
                    outputs = net(inputs)

                    # Compute loss
                    loss = self.loss_function(outputs, targets)

                    # Perform backward pass
                    loss.backward()

                    # Perform optimization
                    optimizer.step()

                    # See results
                    epoch_loss += loss.item()

                if epoch % evaluation_timestep == 0:
                    # Print epoch
                    print(f"Evaluating epoch {epoch+1}")

                    if debug:
                        total, correct, curr_train_loss = self.evaluate(
                            net, trainloader
                        )
                        train_accuracy = 100.0 * (correct / total)
                        total, correct, curr_test_loss = self.evaluate(net, testloader)
                        test_accuracy = 100.0 * (correct / total)
                        print("Epoch Train loss", curr_train_loss)
                        print("Epoch Test loss", curr_test_loss)
                        print("Train accuracy", train_accuracy)
                        print("Test accuracy", test_accuracy)
                        epoch_idx = math.floor(epoch / evaluation_timestep)
                        train_loss[epoch_idx, fold] = curr_train_loss
                        test_loss[epoch_idx, fold] = curr_test_loss
                        train_accuracies[epoch_idx, fold] = train_accuracy
                        test_accuracies[epoch_idx, fold] = test_accuracy

                        self.plot_metrics(
                            train_loss[: epoch_idx + 1, : fold + 1],
                            test_loss[: epoch_idx + 1, : fold + 1],
                            train_accuracies[: epoch_idx + 1, : fold + 1],
                            test_accuracies[: epoch_idx + 1, : fold + 1],
                            ax_train_loss,
                            ax_test_loss,
                            ax_train_accuracy,
                            ax_test_accuracy,
                            evaluation_timestep,
                        )
            # Process is complete.
            print("Training process has finished. Saving trained model.")
            print("Starting testing")
            total, correct, curr_train_loss = self.evaluate(net, trainloader)
            final_train_loss[fold] = curr_train_loss
            final_train_accuracies[fold] = 100.0 * (correct / total)
            total, correct, curr_test_loss = self.evaluate(net, testloader)
            final_test_loss[fold] = curr_test_loss
            final_test_accuracies[fold] = 100.0 * (correct / total)
            print(f"Final train loss for {fold} is {final_train_loss[:fold+1]}")
            print(f"Final test loss for {fold} is {final_test_loss[:fold+1]}")
            print(
                f"Final train accuracy for {fold} is {final_train_accuracies[:fold+1]}"
            )
            print(f"Final test accuracy for {fold} is {final_test_accuracies[:fold+1]}")
            # Print about testing
            if self.save_path is not None:
                # Saving the model
                path = os.path.join(fold_path, "network.pt")
                torch.save(net.state_dict(), path)

                path = os.path.join(fold_path, "train_loader.pt")
                torch.save(trainloader, path)

                path = os.path.join(fold_path, "test_loader.pt")
                torch.save(testloader, path)
            if debug == True:
                interp_tpr_fold, roc_auc_fold = self.compute_one_v_rest_roc(
                    net, dataset.diagnoses_mappings, testloader, interp_fpr
                )
                for class_label in all_tpr.keys():
                    all_tpr[class_label][fold, :] = interp_tpr_fold[class_label]
                    all_auc[class_label][fold] = roc_auc_fold[class_label]

                self.plot_mean_roc(fold, interp_fpr, all_tpr, all_auc, mean_roc_fig)

                # Print accuracy
                print(f"Accuracy for fold {fold}: {final_test_accuracies[fold]} %%")
                print("--------------------------------")

        # Print fold results
        if debug:
            print(f"K-FOLD CROSS VALIDATION RESULTS FOR {self.num_folds} FOLDS")
            print("--------------------------------")
            print("Train loss:")
            for i in range(self.num_folds):
                print(f"Fold {i}: {final_train_loss[i]} %")
            print(f"Average: {np.average(final_train_loss)} %")

            print("Test loss:")
            for i in range(self.num_folds):
                print(f"Fold {i}: {final_test_loss[i]} %")
            print(f"Average: {np.average(final_test_loss)} %")

            print("Train Accuracy:")
            for i in range(self.num_folds):
                print(f"Fold {i}: {final_train_accuracies[i]} %")
            print(f"Average: {np.average(final_train_accuracies)} %")

            print("Test Accuracy:")
            for i in range(self.num_folds):
                print(f"Fold {i}: {final_test_accuracies[i]} %")
            print(f"Average: {np.average(final_test_accuracies)} %")

            print("AUCs:")
            for ndd_class in all_auc.keys():
                print("\t", ndd_class)
                for i in range(self.num_folds):
                    print(f"\t\tFold {i}: {all_auc[ndd_class][i]}")
                print(f"\t\tAverage: {np.average(all_auc[ndd_class])}")
                print("---")

        if self.save_path is not None:
            np.save(os.path.join(self.save_path, "train_loss"), train_loss)
            np.save(os.path.join(self.save_path, "test_loss"), test_loss)
            np.save(os.path.join(self.save_path, "train_accuracy"), train_accuracies)
            np.save(os.path.join(self.save_path, "test_accuracy"), test_accuracies)
            np.save(os.path.join(self.save_path, "all_tpr"), all_tpr)
            np.save(os.path.join(self.save_path, "all_auc"), all_auc)

        if debug:
            plt.show()
        return np.average(final_train_loss), np.average(final_test_accuracies)

    def compute_one_v_rest_roc(self, lstm, label_mappings, testloader, interp_fpr):
        # global test_y
        interp_tpr = dict()
        roc_auc = dict()

        lstm.eval()
        test_y, y_score = self.predict_probs(lstm, testloader)
        test_y = test_y.detach().cpu().numpy()
        y_score = y_score.detach().cpu().numpy()

        for ndd_class in label_mappings.keys():
            i = label_mappings[ndd_class]
            curr_fpr, curr_tpr, _ = roc_curve(test_y, y_score[:, i], pos_label=i)
            interp_tpr[ndd_class] = np.interp(interp_fpr, curr_fpr, curr_tpr)
            interp_tpr[ndd_class][0] = 0.0
            roc_auc[ndd_class] = auc(interp_fpr, interp_tpr[ndd_class])
            # print("\tfpr for ", i, fpr[i])
            # print("\ttpr for ", i, tpr[i])
            print("\tROC for ", ndd_class, ":", i, roc_auc[ndd_class])
            print("--------------------")
        return interp_tpr, roc_auc

    def plot_mean_roc(self, up_to_fold, interp_fpr, interp_tpr, all_auc, fig):
        plt.figure(fig)
        fig.clear()
        colors = ["blue", "red", "green", "black"]
        for i, class_label in enumerate(interp_tpr.keys()):
            curr_tpr = interp_tpr[class_label][: up_to_fold + 1, :]
            mean_tpr = np.mean(curr_tpr, 0)
            std_tpr = np.std(curr_tpr, 0)

            curr_auc = all_auc[class_label][: up_to_fold + 1]
            mean_auc = np.mean(curr_auc)
            std_auc = np.std(curr_auc)
            plt.plot(
                interp_fpr,
                mean_tpr,
                color=colors[i],
                label=r"ROC curve for %s (area = %0.2f $\pm$ %0.2f)"
                % (class_label, mean_auc, std_auc),
            )
            plt.fill_between(
                interp_fpr, mean_tpr + std_tpr, mean_tpr - std_tpr, alpha=0.5
            )

        plt.plot([0, 1], [0, 1], "k--", linewidth=4)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=20)
        plt.ylabel("True Positive Rate", fontsize=20)
        plt.title("Receiver operating characteristic for multi-class data", fontsize=30)
        plt.legend(loc="lower right", prop={"size": 20})
        plt.gca().tick_params(axis="both", labelsize=15)
        plt.pause(0.005)
