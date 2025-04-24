import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
import matplotlib.pyplot as plt
import os
import math
import pickle
from .move_datasets import MovementDataset, SimpleDataset
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import shutil

# Reset the weights of a network to their initial values
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
        y_pred = self.linear(dropout_op)
        return y_pred

# Class that creates and trains a network
class LSTMNetwork:
    def __init__(
        self,
        num_folds,
        input_dim=9,
        hidden_dim_list=[10, 20],
        output_dim=4,
        save_path="../data/models/",
        learning_rate_list=[0.001, 0.0025, 0.005],
        weight_decay_list=[0.001, 0.005, 0.01, 0.05],
        num_lstm_layers_list=[1, 2],
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
        self.hidden_dim_list = hidden_dim_list
        self.output_dim = output_dim
        self.save_path = save_path
        self.learning_rate_list = learning_rate_list
        self.weight_decay_list = weight_decay_list
        self.num_lstm_layers_list = num_lstm_layers_list
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

    # Helper function to display a sequence of a metric over epochs
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

    # Function to plot all metrics over epochs
    def plot_metrics(
        self,
        evaluation_str,
        train_loss,
        evaluation_loss,
        train_accuracy,
        evaluation_accuracy,
        ax_train_loss,
        ax_evaluation_loss,
        ax_train_accuracy,
        ax_evaluation_accuracy,
        evaluation_timestep,
    ):
        self.plot_single_metric(ax_train_loss, train_loss, evaluation_timestep)
        ax_train_loss.set_title("Training Loss")
        ax_train_loss.set(xlabel="# of Epochs", ylabel="Loss")

        self.plot_single_metric(
            ax_evaluation_loss, evaluation_loss, evaluation_timestep
        )
        ax_evaluation_loss.set_title(evaluation_str + " Loss")
        ax_evaluation_loss.set(xlabel="# of Epochs", ylabel="Loss")

        self.plot_single_metric(ax_train_accuracy, train_accuracy, evaluation_timestep)
        ax_train_accuracy.set_ylim([0, 100.00])
        ax_train_accuracy.set_title("Training Accuracy")
        ax_train_accuracy.set(xlabel="# of Epochs", ylabel="Accuracy")
        ax_train_accuracy.set_aspect("auto")

        self.plot_single_metric(
            ax_evaluation_accuracy, evaluation_accuracy, evaluation_timestep
        )
        ax_evaluation_accuracy.set_title(evaluation_str + " Accuracy")
        ax_evaluation_accuracy.set(xlabel="# of Epochs", ylabel="Accuracy")
        ax_evaluation_accuracy.set_aspect("auto")

        plt.pause(0.01)

    # Function to evaluate the model on a dataset
    # Returns the loss on that dataset, the confusion matrix, and a classification report
    def evaluate(self, net, dataLoader, compute_report=False):
        # Evaluationfor this fold
        net.eval()

        desired_labels, predictions, total_loss = self.predict(net, dataLoader)

        # Set total and correct
        _, predicted = torch.max(predictions, 1)

        cm = confusion_matrix(
            desired_labels.detach().cpu().numpy(), predicted.detach().cpu().numpy()
        )
        prediction_report = None
        if compute_report:
            prediction_report = classification_report(
                desired_labels.detach().cpu().numpy(),
                predicted.detach().cpu().numpy(),
            )
        return total_loss, cm, prediction_report

    # Function that trains the model on a single training and evaluation set
    # This is useful for training one fold, or for training on the entire dataset
    def trainAndEvaluateOnDatasets(
        self,
        fold,
        save_path,
        num_epochs,
        early_stopping_obj,
        evaluation_str,
        trainloader,
        evaluationloader,
        evaluation_timestep=10,
        debug=True,
    ):
        net = Net(
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
            self.batch_size,
            self.num_lstm_layers,
            self.dropout,
        )
        net.to(self.device)
        net.apply(reset_weights)
        optimizer = torch.optim.Adam(
            net.parameters(), weight_decay=self.weight_decay, lr=self.learning_rate
        )

        if early_stopping_obj is not None:
            early_stopping_obj.reset()

        if debug:
            num_evaluations = math.ceil(num_epochs / evaluation_timestep)
            train_accuracies = np.zeros((num_evaluations, 1))
            train_loss = np.zeros((num_evaluations, 1))
            evaluation_accuracies = np.zeros((num_evaluations, 1))
            evaluation_loss = np.zeros((num_evaluations, 1))

            all_tpr = {}
            num_roc_points = 500
            interp_fpr = np.linspace(0, 1, num_roc_points)
            all_auc = {}
            for ndd_class in trainloader.dataset.diagnoses_mappings.keys():
                all_tpr[ndd_class] = np.zeros(num_roc_points)
                all_auc[ndd_class] = 0

            training_fig = plt.figure()
            ax_train_loss = training_fig.add_subplot(221)
            ax_train_loss.set_title("Training Loss", fontsize=15)

            ax_evaluation_loss = training_fig.add_subplot(222, sharex=ax_train_loss)
            ax_evaluation_loss.set_title(evaluation_str + " Loss", fontsize=15)

            ax_train_accuracy = training_fig.add_subplot(223, sharex=ax_train_loss)
            ax_train_accuracy.set_title("Training Accuracy", fontsize=15)
            ax_train_accuracy.set_ylim([0, 100])

            ax_evaluation_accuracy = training_fig.add_subplot(
                224, sharex=ax_train_loss, sharey=ax_train_accuracy
            )
            ax_evaluation_accuracy.set_title(evaluation_str + " Accuracy", fontsize=15)

            roc_fig = plt.figure()

        for epoch in range(0, num_epochs):
            epoch_loss = 0
            for i, data in enumerate(trainloader):
                # Get inputs
                inputs, targets = data
                inputs = inputs.transpose(0, 2).transpose(1, 2)
                targets = targets.view(-1)
                net.train()
                # Zero the gradients
                optimizer.zero_grad()
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
                if debug:
                    curr_train_loss, train_cm, _ = self.evaluate(net, trainloader)
                    curr_train_accuracy = 100.0 * (train_cm.trace() / train_cm.sum())
                    curr_evaluation_loss, evaluation_cm, _ = self.evaluate(
                        net, evaluationloader
                    )
                    curr_evaluation_accuracy = 100.0 * (
                        evaluation_cm.trace() / evaluation_cm.sum()
                    )
                    epoch_idx = math.floor(epoch / evaluation_timestep)
                    train_loss[epoch_idx] = curr_train_loss
                    evaluation_loss[epoch_idx] = curr_evaluation_loss
                    train_accuracies[epoch_idx] = curr_train_accuracy
                    evaluation_accuracies[epoch_idx] = curr_evaluation_accuracy

                    self.plot_metrics(
                        evaluation_str,
                        train_loss[: epoch_idx + 1],
                        evaluation_loss[: epoch_idx + 1],
                        train_accuracies[: epoch_idx + 1],
                        evaluation_accuracies[: epoch_idx + 1],
                        ax_train_loss,
                        ax_evaluation_loss,
                        ax_train_accuracy,
                        ax_evaluation_accuracy,
                        evaluation_timestep,
                    )

                is_stop_early = False
                if early_stopping_obj is not None:
                    early_stopping_obj(curr_evaluation_loss, net)
                    is_stop_early = early_stopping_obj.is_stop_early
                if is_stop_early:
                    break

        # Process is complete.
        print(f"Training process has finished for fold {fold}. Saving trained model.")
        print("Starting validation")
        final_train_loss, final_train_cm, _ = self.evaluate(net, trainloader)
        final_train_accuracy = 100.0 * (final_train_cm.trace() / final_train_cm.sum())
        final_evaluation_loss, final_evaluation_cm, final_evaluation_report = (
            self.evaluate(net, evaluationloader, compute_report=True)
        )
        final_evaluation_accuracy = 100.0 * (
            final_evaluation_cm.trace() / final_evaluation_cm.sum()
        )
        print(f"Final train loss for fold {fold} is {final_train_loss}")
        print(f"Final evaluation loss for fold {fold} is {final_evaluation_loss}")
        print(f"Final train accuracy for fold {fold} is {final_train_accuracy}")
        print(
            f"Final evaluation accuracy for fold {fold} is {final_evaluation_accuracy}"
        )
        print(f"Final confusion matrix for {evaluation_str}\n", final_evaluation_cm)
        print(
            f"Final classification report for {evaluation_str}\n",
            final_evaluation_report,
        )
        if debug:

            interp_tpr_fold, roc_auc_fold = self.compute_one_v_rest_roc(
                net,
                trainloader.dataset.diagnoses_mappings,
                evaluationloader,
                interp_fpr,
            )
            for class_label in all_tpr.keys():
                all_tpr[class_label] = interp_tpr_fold[class_label]
                all_auc[class_label] = roc_auc_fold[class_label]

            self.plot_roc(interp_fpr, all_tpr, all_auc, roc_fig)

        if save_path is not None:
            fold_path = os.path.join(save_path, f"model-fold-{fold}")

            if not os.path.exists(fold_path):
                os.makedirs(fold_path)
            path = os.path.join(fold_path, "network.pt")
            torch.save(
                {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "output_dim": self.output_dim,
                    "batch_size": self.batch_size,
                    "num_lstm_layers": self.num_lstm_layers,
                    "dropout": self.dropout,
                    "model_state_dict": net.state_dict(),
                },
                path,
            )
            path = os.path.join(fold_path, "train_loader.pt")
            torch.save(trainloader, path)
            path = os.path.join(fold_path, "evaluation_loader.pt")
            torch.save(evaluationloader, path)

            np.save(os.path.join(fold_path, "train_loss"), train_loss)
            np.save(os.path.join(fold_path, "evaluation_loss"), evaluation_loss)
            np.save(os.path.join(fold_path, "train_accuracy"), train_accuracies)
            np.save(
                os.path.join(fold_path, "evaluation_accuracy"), evaluation_accuracies
            )
            np.save(os.path.join(fold_path, "tpr"), all_tpr)
            np.save(os.path.join(fold_path, "auc"), all_auc)
            np.save(os.path.join(fold_path, "final_train_cm"), final_train_cm)
            np.save(os.path.join(fold_path, "final_evaluation_cm"), final_evaluation_cm)
            with open(
                os.path.join(fold_path, "final_evaluation_cm_report.txt"), "w"
            ) as f:
                f.write(final_evaluation_report)

            cm_labels = np.full(len(final_evaluation_cm), None)
            cm_labels[list(trainloader.dataset.diagnoses_mappings.values())] = list(
                trainloader.dataset.diagnoses_mappings.keys()
            )
            disp = ConfusionMatrixDisplay(
                confusion_matrix=final_evaluation_cm, display_labels=cm_labels
            )
            disp.plot()

        if debug:
            plt.close("all")
        return (
            all_auc,
            final_train_loss,
            final_evaluation_loss,
            final_train_accuracy,
            final_evaluation_accuracy,
            epoch,
        )

    # Train the network using K-Fold Cross Validaiton on the training set
    # Then train on entire training set and evaluate on test set
    def train(
        self,
        src_dataset: MovementDataset,
        num_epochs=500,
        test_ratio=0.2,
        early_stopping_obj=None,
        debug=True,
    ):
        if not self.num_folds > 0:
            print("Cannot train without hyperparameter tuning")
            return
        training_dataset, testing_dataset = train_test_split(
            src_dataset, test_size=test_ratio, stratify=src_dataset.y
        )
        testing_dataset_X, testing_dataset_y = map(list, zip(*testing_dataset))
        testing_dataset = SimpleDataset(
            testing_dataset_X, testing_dataset_y, src_dataset.diagnoses_mappings
        )
        testloader = torch.utils.data.DataLoader(
            testing_dataset, batch_size=self.batch_size
        )

        training_dataset_X, training_dataset_y = map(list, zip(*training_dataset))
        training_dataset = SimpleDataset(
            training_dataset_X, training_dataset_y, src_dataset.diagnoses_mappings
        )
        evaluation_timestep = 10

        hyperparams, max_epochs = self.hyperparameter_search(
            training_dataset,
            num_epochs,
            early_stopping_obj,
            evaluation_timestep,
            debug,
        )

        print("Best hyperparams", hyperparams)
        self.hidden_dim_list = hyperparams["hidden_dim"]
        self.learning_rate_list = hyperparams["learning_rate"]
        self.weight_decay_list = hyperparams["weight_decay"]
        self.num_lstm_layers_list = hyperparams["num_lstm_layers"]

        if max_epochs % evaluation_timestep != 0:
            max_epochs = (
                int(evaluation_timestep - max_epochs % evaluation_timestep) + max_epochs
            )
        print("Running final training for ", max_epochs, "epochs")
        self.trainFinalModel(
            training_dataset,
            testloader,
            max_epochs,
            evaluation_timestep,
            debug,
        )

    # Perform Hyperparameter search on the training set
    def hyperparameter_search(
        self,
        training_dataset,
        num_epochs,
        early_stopping_obj,
        evaluation_timestep,
        debug=True,
    ):
        best_hyperparams = None
        best_performance = 0
        best_max_epochs = None
        for hidden_dim in self.hidden_dim_list:
            for learning_rate in self.learning_rate_list:
                for weight_decay in self.weight_decay_list:
                    for num_lstm_layers in self.num_lstm_layers_list:
                        print(
                            f"Training with hidden_dim={hidden_dim}, learning_rate={learning_rate}, weight_decay={weight_decay}, num_lstm_layers={num_lstm_layers}"
                        )
                        self.hidden_dim = hidden_dim
                        self.learning_rate = learning_rate
                        self.weight_decay = weight_decay
                        self.num_lstm_layers = num_lstm_layers

                        # Train the network using K-Fold Cross Validation on the training set
                        max_epochs, avg_auc = self.trainKFold(
                            training_dataset,
                            num_epochs,
                            early_stopping_obj,
                            evaluation_timestep,
                            debug,
                        )
                        if avg_auc > best_performance:
                            best_performance = avg_auc
                            best_max_epochs = max_epochs
                            best_hyperparams = {
                                "hidden_dim": hidden_dim,
                                "learning_rate": learning_rate,
                                "weight_decay": weight_decay,
                                "num_lstm_layers": num_lstm_layers,
                            }

                            if os.path.exists(os.path.join(self.save_path, "Best")):
                                shutil.rmtree(os.path.join(self.save_path, "Best"))
                            os.rename(
                                os.path.join(self.save_path, "K-Fold"),
                                os.path.join(self.save_path, "Best"),
                            )
        if os.path.exists(os.path.join(self.save_path, "K-Fold")):
            shutil.rmtree(os.path.join(self.save_path, "K-Fold"))
        os.rename(os.path.join(self.save_path, "Best"), os.path.join(self.save_path, "K-Fold"))
        return best_hyperparams, best_max_epochs
    # Train the network on the entire training set and evaluate on test set
    def trainFinalModel(
        self,
        training_dataset,
        testloader,
        num_epochs,
        evaluation_timestep,
        debug,
    ):
        trainloader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        (
            auc,
            final_train_loss,
            final_test_loss,
            final_train_accuracy,
            final_test_accuracy,
            _,
        ) = self.trainAndEvaluateOnDatasets(
            0,
            os.path.join(self.save_path, "Final"),
            num_epochs,
            None,
            "Test",
            trainloader,
            testloader,
            evaluation_timestep,
            debug,
        )
        if debug:
            print(f"Train loss: {final_train_loss}")
            print(f"Test loss: {final_test_loss}")
            print(f"Train Accuracy: {final_train_accuracy}")
            print(f"Test Accuracy: {final_test_accuracy}")
            print("AUCs:")
            for ndd_class in auc.keys():
                print("\t", ndd_class, auc[ndd_class])
            print("--------------------------------")

    # Function that trains the network using K-Fold Cross Validation on the training set
    def trainKFold(
        self,
        training_dataset,
        num_epochs,
        early_stopping_obj,
        evaluation_timestep,
        debug,
    ):
        if debug:
            fold_wise_auc = {}
            for class_label in training_dataset.diagnoses_mappings.keys():
                fold_wise_auc[class_label] = {}
            fold_wise_final_train_loss = {}
            fold_wise_final_validation_loss = {}
            fold_wise_final_train_accuracy = {}
            fold_wise_final_validation_accuracy = {}

        training_dataset_X = training_dataset.X
        training_dataset_y_np = [x.detach().cpu().numpy() for x in training_dataset.y]

        kfold = StratifiedKFold(n_splits=self.num_folds, shuffle=True)
        splits = kfold.split(training_dataset_X, training_dataset_y_np)

        max_epochs = 0
        for fold, (train_ids, validation_ids) in enumerate(splits):
            # for fold in range(self.num_folds):
            print(f"FOLD {fold}")
            print("--------------------------------")
            print("Num train", len(train_ids))
            print("Num Validation", len(validation_ids))

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)
            # Define data loaders for training and validationing data in this fold
            trainloader = torch.utils.data.DataLoader(
                training_dataset, batch_size=self.batch_size, sampler=train_subsampler
            )
            validationloader = torch.utils.data.DataLoader(
                training_dataset,
                batch_size=self.batch_size,
                sampler=validation_subsampler,
            )

            (
                fold_auc,
                fold_final_train_loss,
                fold_final_validation_loss,
                fold_final_train_accuracy,
                fold_final_validation_accuracy,
                fold_epochs,
            ) = self.trainAndEvaluateOnDatasets(
                fold,
                os.path.join(self.save_path, "K-Fold"),
                num_epochs,
                early_stopping_obj,
                "Validation",
                trainloader,
                validationloader,
                evaluation_timestep,
                debug,
            )
            if fold_epochs > max_epochs:
                max_epochs = fold_epochs

            for ndd_class in fold_auc.keys():
                fold_wise_auc[ndd_class][fold] = fold_auc[ndd_class]
            fold_wise_final_train_loss[fold] = fold_final_train_loss
            fold_wise_final_validation_loss[fold] = fold_final_validation_loss
            fold_wise_final_train_accuracy[fold] = fold_final_train_accuracy
            fold_wise_final_validation_accuracy[fold] = fold_final_validation_accuracy

            # Print accuracy
            print(f"Accuracy for fold {fold}: {fold_final_validation_accuracy} %%")
            print("--------------------------------")

        # Print fold results
        if debug:
            print(f"K-FOLD CROSS VALIDATION RESULTS FOR {self.num_folds} FOLDS")
            print("--------------------------------")
            print("Train loss:")
            for i in range(self.num_folds):
                print(f"Fold {i}: {fold_wise_final_train_loss[i]} %")
            print(f"Average: {np.average(list(fold_wise_final_train_loss.values()))} %")

            print("validation loss:")
            for i in range(self.num_folds):
                print(f"Fold {i}: {fold_wise_final_validation_loss[i]} %")
            print(
                f"Average: {np.average(list(fold_wise_final_validation_loss.values()))} %"
            )

            print("Train Accuracy:")
            for i in range(self.num_folds):
                print(f"Fold {i}: {fold_wise_final_train_accuracy[i]} %")
            print(
                f"Average: {np.average(list(fold_wise_final_train_accuracy.values()))} %"
            )

            print("validation Accuracy:")
            for i in range(self.num_folds):
                print(f"Fold {i}: {fold_wise_final_validation_accuracy[i]} %")
            print(
                f"Average: {np.average(list(fold_wise_final_validation_accuracy.values()))} %"
            )

            print("AUCs:")
            for ndd_class in fold_wise_auc.keys():
                print("\t", ndd_class)
                for i in range(self.num_folds):
                    print(f"\t\tFold {i}: {fold_wise_auc[ndd_class][i]}")
                print(
                    f"\t\tAverage: {np.average(list(fold_wise_auc[ndd_class].values()))}"
                )
                print("---")

        mean_macro_average_auc = sum(
            v for inner in fold_wise_auc.values() for v in inner.values()
        ) / sum(len(inner) for inner in fold_wise_auc.values())

        return max_epochs, mean_macro_average_auc

    # Compute ROC for each class
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

    # Plot ROC curves
    def plot_roc(self, interp_fpr, interp_tpr, all_auc, fig):
        plt.figure(fig)
        fig.clear()
        colors = ["blue", "red", "green", "black"]
        for i, class_label in enumerate(interp_tpr.keys()):
            plt.plot(
                interp_fpr,
                interp_tpr[class_label],
                color=colors[i],
                label=r"ROC curve for %s (area = %0.2f)"
                % (class_label, all_auc[class_label]),
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
