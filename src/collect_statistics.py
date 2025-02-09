import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, ConfusionMatrixDisplay
import argparse
import torch

title_size = 20
label_size = 15
tick_size = 12


def get_roc_metrics(file_path):
    if os.path.exists(file_path) == False:
        return {}
    roc_data = np.load(file_path, allow_pickle=True).item()
    return roc_data


def get_KFold_roc_metrics(experiment_path, condition_list):
    final_aucs = {}
    mean_aucs = {}
    std_aucs = {}
    for condition in condition_list:
        final_aucs[condition] = []
    fold_folders = [
        x.name
        for x in os.scandir(os.path.join(experiment_path, "K-Fold"))
        if x.is_dir()
    ]
    fold_folders.sort()
    for fold_folder in fold_folders:
        fold_aucs = get_roc_metrics(
            os.path.join(experiment_path, "K-Fold", fold_folder, "auc.npy")
        )
        for condition in fold_aucs:
            final_aucs[condition].append(fold_aucs[condition])

    for condition in final_aucs:
        mean_aucs[condition] = np.mean(final_aucs[condition])
        std_aucs[condition] = np.std(final_aucs[condition])
    return final_aucs.keys(), mean_aucs, std_aucs


def get_final_accuracy(file_path):
    if os.path.exists(file_path) == False:
        return 0
    accuracy = np.load(file_path)
    non_zero_indices = np.nonzero(accuracy)
    non_zero_indices = np.vstack(non_zero_indices).T
    return accuracy[non_zero_indices[-1, 0]].item()


def get_KFold_accuracy_metrics(experiment_path):
    final_validation_accuracies = []
    fold_folders = [
        x.name
        for x in os.scandir(os.path.join(experiment_path, "K-Fold"))
        if x.is_dir()
    ]
    fold_folders.sort()
    for fold_folder in fold_folders:
        final_validation_accuracies.append(
            get_final_accuracy(
                os.path.join(
                    experiment_path, "K-Fold", fold_folder, "evaluation_accuracy.npy"
                )
            )
        )
    mean, std = np.mean(final_validation_accuracies), np.std(
        final_validation_accuracies
    )
    return mean, std


def plot_single_metric(ax, metric, evaluation_timestep, title, xlabel, ylabel, fold):
    metric = np.array(metric)
    metric = np.where(np.isclose(metric, 0), np.nan, metric)
    num_epochs = metric.shape[0]
    if num_epochs == 0:
        return
    x_range = np.arange(num_epochs) * evaluation_timestep
    # ax.clear()

    if metric.ndim > 1 and metric.shape[1] > 1:
        mean_metric = np.nanmean(metric, axis=1)
        ax.plot(x_range, mean_metric, label="Mean")
        std_metric = 2 * np.nanstd(metric, axis=1)
        ax.fill_between(
            x_range, mean_metric - std_metric, mean_metric + std_metric, alpha=0.5
        )

        # ax.legend()
        ax.set_title(title, fontsize=title_size)
        ax.set_xlabel(xlabel, fontsize=label_size)
        ax.set_ylabel(ylabel, fontsize=label_size)
        ax.tick_params(axis="both", labelsize=tick_size)
    else:
        ax.plot(x_range, metric, label=fold)
        ax.set_title(title, fontsize=title_size)
        ax.set_xlabel(xlabel, fontsize=label_size)
        ax.set_ylabel(ylabel, fontsize=label_size)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.legend()


def redraw_training_data(instances_path, evaluation_str, output_folder):
    training_fig = plt.figure()
    ax_train_loss = training_fig.add_subplot(221)

    ax_test_loss = training_fig.add_subplot(222, sharex=ax_train_loss)

    ax_train_accuracy = training_fig.add_subplot(223, sharex=ax_train_loss)

    ax_test_accuracy = training_fig.add_subplot(
        224, sharex=ax_train_loss, sharey=ax_train_accuracy
    )
    evaluation_timestep = 10

    fold_folders = [f.name for f in os.scandir(instances_path) if f.is_dir()]
    fold_folders.sort()

    for fold in fold_folders:
        fold_path = os.path.join(instances_path, fold)
        fold_train_loss = np.load(os.path.join(fold_path, "train_loss.npy"))
        fold_evaluation_loss = np.load(os.path.join(fold_path, "evaluation_loss.npy"))
        fold_train_accuracy = np.load(os.path.join(fold_path, "train_accuracy.npy"))
        fold_evaluation_accuracy = np.load(
            os.path.join(fold_path, "evaluation_accuracy.npy")
        )

        if len(fold_folders) == 1:
            fold = None

        plot_single_metric(
            ax_train_loss,
            fold_train_loss,
            evaluation_timestep,
            "Training Loss",
            "# of Epochs",
            "Loss",
            fold=fold,
        )

        plot_single_metric(
            ax_test_loss,
            fold_evaluation_loss,
            evaluation_timestep,
            evaluation_str + " Loss",
            "# of Epochs",
            "Loss",
            fold=fold,
        )

        plot_single_metric(
            ax_train_accuracy,
            fold_train_accuracy,
            evaluation_timestep,
            "Training Accuracy",
            "# of Epochs",
            "Accuracy",
            fold=fold,
        )

        plot_single_metric(
            ax_test_accuracy,
            fold_evaluation_accuracy,
            evaluation_timestep,
            evaluation_str + " Accuracy",
            "# of Epochs",
            "Accuracy",
            fold=fold,
        )

    ax_train_accuracy.set_ylim([0, 100.00])

    training_fig.subplots_adjust(wspace=0.12, hspace=0.32)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    mng.resize(*mng.window.maxsize())
    plt.pause(0.5)

    training_fig.savefig(
        os.path.join(output_folder, "training_results.svg"), bbox_inches="tight"
    )
    plt.close(training_fig)


def redraw_roc(instances_path, output_folder, condition_list):
    num_roc_points = 500

    fold_folders = [f.name for f in os.scandir(instances_path) if f.is_dir()]
    fold_folders.sort()

    tprs = {}
    for condition in condition_list:
        tprs[condition] = np.zeros((len(fold_folders), num_roc_points))
    for i, fold in enumerate(fold_folders):
        fold_path = os.path.join(instances_path, fold)
        fold_tpr = np.load(os.path.join(fold_path, "tpr.npy"), allow_pickle=True).item()
        for condition in fold_tpr:
            tprs[condition][i, :] = fold_tpr[condition]

    redraw_all_roc(
        num_roc_points, tprs, output_folder, len(fold_folders) > 1
    )
    redraw_individual_roc(
        num_roc_points, tprs, output_folder, len(fold_folders) > 1
    )


def redraw_all_roc(num_roc_points, tprs, output_folder, with_std=False):
    fpr = np.linspace(0, 1, num_roc_points)

    colors = ["blue", "red", "green", "black", "orange"]

    without_std_fig = plt.figure()
    without_std_ax = without_std_fig.gca()
    if with_std:
        with_std_fig = plt.figure()
        with_std_ax = with_std_fig.gca()
    for i, ndd in enumerate(tprs.keys()):
        ndd_tpr = tprs[ndd]

        if with_std:
            if ndd_tpr.ndim <= 1 or ndd_tpr.shape[0] <= 1:
                continue

            mean_tpr = np.mean(ndd_tpr, 0)
            std_tpr = np.std(ndd_tpr, 0)

            fold_aucs = []
            for fold in range(ndd_tpr.shape[0]):
                fold_aucs.append(auc(fpr, ndd_tpr[fold, :]))

            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)

            without_std_ax.plot(
                fpr,
                mean_tpr,
                color=colors[i],
                label=r"ROC curve for %s (area = %0.2f $\pm$ %0.2f)"
                % (ndd, mean_auc, std_auc),
                linewidth=4,
            )

            with_std_ax.plot(
                fpr,
                mean_tpr,
                color=colors[i],
                label=r"ROC curve for %s (area = %0.2f $\pm$ %0.2f)"
                % (ndd, mean_auc, std_auc),
                linewidth=2,
            )
            with_std_ax.fill_between(
                fpr, mean_tpr + std_tpr, mean_tpr - std_tpr, alpha=0.5
            )
            fig_list = [
                (without_std_fig, without_std_ax, "all_roc_without_std.svg"),
                (with_std_fig, with_std_ax, "all_roc_with_std.svg"),
            ]

        else:
            ndd_tpr = ndd_tpr[0, :]
            fold_auc = auc(fpr, ndd_tpr)
            without_std_ax.plot(
                fpr,
                ndd_tpr,
                color=colors[i],
                label=r"ROC curve for %s (area = %0.2f)" % (ndd, fold_auc),
                linewidth=4,
            )
            fig_list = [(without_std_fig, without_std_ax, "all_roc_without_std.svg")]

    for fig, ax, filename in fig_list:
        ax.plot([0, 1], [0, 1], "k--", linewidth=4)
        ax.set_xlim(-0.05, 1.0)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("False Positive Rate", fontsize=1.5 * label_size)
        ax.set_ylabel("True Positive Rate", fontsize=1.5 * label_size)
        ax.set_title(
            "Receiver operating characteristic for multi-class data",
            fontsize=1.5 * title_size,
        )
        ax.legend(loc="lower right", prop={"size": 20})
        ax.tick_params(axis="both", labelsize=1.5 * tick_size)

        mng = fig.canvas.manager
        mng.full_screen_toggle()
        mng.resize(*mng.window.maxsize())
        plt.pause(0.5)

        fig.savefig(os.path.join(output_folder, filename), bbox_inches="tight")

        plt.close(fig)


def redraw_individual_roc(
    num_roc_points, tprs, output_folder, with_std=False
):
    fpr = np.linspace(0, 1, num_roc_points)

    for ndd in tprs.keys():
        ndd_tpr = tprs[ndd]
        if with_std:
            mean_tpr = np.mean(ndd_tpr, 0)
            std_tpr = np.std(ndd_tpr, 0)
            fold_aucs = []
            for fold in range(ndd_tpr.shape[0]):
                fold_aucs.append(auc(fpr, ndd_tpr[fold, :]))

            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)

            fig = plt.figure()
            ax = fig.gca()
            ax.plot(
                fpr,
                mean_tpr,
                label=r"Area = %0.2f $\pm$ %0.2f" % (mean_auc, std_auc),
                linewidth=4,
            )
            ax.fill_between(fpr, mean_tpr + std_tpr, mean_tpr - std_tpr, alpha=0.5)

        else:
            ndd_tpr = ndd_tpr[0, :]
            fold_auc = auc(fpr, ndd_tpr)
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(fpr, ndd_tpr, label=r"Area = %0.2f" % (fold_auc), linewidth=4)

        ax.plot([0, 1], [0, 1], "k--", linewidth=4)
        ax.set_xlim(-0.05, 1.0)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("False Positive Rate", fontsize=1.5 * label_size)
        ax.set_ylabel("True Positive Rate", fontsize=1.5 * label_size)
        ax.set_title(
            f"Receiver operating characteristic for {ndd} vs Rest",
            fontsize=1.5 * title_size,
        )
        ax.legend(loc="lower right", prop={"size": 20})
        ax.tick_params(axis="both", labelsize=1.5 * tick_size)

        mng = fig.canvas.manager
        mng.full_screen_toggle()
        mng.resize(*mng.window.maxsize())
        plt.pause(0.5)
        fig.savefig(
            os.path.join(output_folder, f"roc_{ndd}_vs_rest.svg"), bbox_inches="tight"
        )
        plt.close(fig)


def redraw_figures(experiment_path, output_folder, condition_list):
    plt.ioff()
    redraw_data(
        os.path.join(experiment_path, "K-Fold"),
        "Validation",
        os.path.join(output_folder, "K-Fold"),
        condition_list,
    )
    redraw_data(
        os.path.join(experiment_path, "Final"),
        "Test",
        os.path.join(output_folder, "Final"),
        condition_list,
    )


def redraw_data(instances_path, evaluation_str, output_folder, condition_list):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    redraw_training_data(instances_path, evaluation_str, output_folder)
    redraw_roc(instances_path, output_folder, condition_list)
    plot_confusion_matrix(instances_path, evaluation_str, output_folder)


def plot_confusion_matrix(instances_path, evaluation_str, output_folder):
    fold_folders = [f.name for f in os.scandir(instances_path) if f.is_dir()]
    fold_folders.sort()

    for fold in fold_folders:
        fold_path = os.path.join(instances_path, fold)
        confusion_matrix = np.load(os.path.join(fold_path, "final_evaluation_cm.npy"))

        dataloader = torch.load(
            os.path.join(fold_path, "evaluation_loader.pt"),
            map_location=torch.device("cpu"),
        )
        labels = np.full(len(confusion_matrix), None)
        labels[list(dataloader.dataset.diagnoses_mappings.values())] = list(
            dataloader.dataset.diagnoses_mappings.keys()
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix, display_labels=labels
        )
        disp.plot()
        disp.figure_.canvas.manager.full_screen_toggle()
        plt.pause(0.5)
        if os.path.exists(os.path.join(output_folder, fold)) == False:
            os.makedirs(os.path.join(output_folder, fold))
        disp.figure_.savefig(
            os.path.join(output_folder, fold, f"{evaluation_str}-confusion_matrix.svg")
        )
        plt.close(disp.figure_)


def collect_results(data_path, draw_plots=True):
    MEAN_KEY = "Mean"
    STD_KEY = "Stdev"

    condition_list = ["ASD", "ADHD", "ASD+ADHD", "NT"]
    signal_map = {
        "linear_accel": "Linear Accel",
        "rpy_dot": "RPY_dot",
        "rpy": "RPY",
        "linear_accel_and_rpy_dot": "Linear Accel + RPY Dot",
        "linear_accel_and_rpy": "Linear Accel + RPY",
        "rpy_dot_and_rpy": "RPY dot + RPY",
        "all": "All",
    }
    output_figure_path = os.path.join(data_path, "figures")
    if not os.path.exists(output_figure_path):
        os.makedirs(output_figure_path)
    validation_accuracy_data = {}
    validation_roc_data = {}

    test_accuracy_data = {}
    test_roc_data = {}

    for signal in signal_map:
        signal_path = os.path.join(data_path, signal)
        if os.path.isdir(signal_path) == False:
            continue
        if signal_path == output_figure_path:
            continue
        print(signal)
        signal_key = signal_map[signal]
        validation_accuracy_data[signal_key] = {}
        validation_roc_data[signal_key] = {}
        test_accuracy_data[signal_key] = None
        test_roc_data[signal_key] = {}

        validation_accuracy_data[signal_key][MEAN_KEY] = None
        validation_accuracy_data[signal_key][STD_KEY] = None


        for ndd_key in condition_list:
            validation_roc_data[signal_key][
                (ndd_key, MEAN_KEY)
            ] = None
            validation_roc_data[signal_key][
                (ndd_key, STD_KEY)
            ] = None

            test_roc_data[signal_key][ndd_key] = None

        experiment_path = signal_path

        mean, std = get_KFold_accuracy_metrics(experiment_path)
        validation_accuracy_data[signal_key][MEAN_KEY] = mean
        validation_accuracy_data[signal_key][STD_KEY] = std

        ndds, roc_means, roc_stds = get_KFold_roc_metrics(
            experiment_path, condition_list
        )
        for ndd_key in ndds:
            validation_roc_data[signal_key][ndd_key, MEAN_KEY] = (
                roc_means[ndd_key]
            )
            validation_roc_data[signal_key][(ndd_key, STD_KEY)] = (
                roc_stds[ndd_key]
            )
        figure_path = os.path.join(output_figure_path, signal)
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        if draw_plots:
            redraw_figures(experiment_path, figure_path, condition_list)

        test_accuracy_file_path = os.path.join(
            experiment_path, "Final", "model-fold-0", "evaluation_accuracy.npy"
        )

        test_accuracy_data[signal_key] = get_final_accuracy(
            test_accuracy_file_path
        )
        test_roc_file_path = os.path.join(
            experiment_path, "Final", "model-fold-0", "auc.npy"
        )
        test_roc_src = get_roc_metrics(test_roc_file_path)
        for ndd_key in test_roc_src:
            test_roc_data[signal_key][ndd_key] = test_roc_src[ndd_key]

    validation_accuracy_df = pd.DataFrame(validation_accuracy_data)
    validation_accuracy_df = validation_accuracy_df.transpose()
    
    validation_roc_df = pd.DataFrame(validation_roc_data)
    validation_roc_df = validation_roc_df.transpose()
    
    test_accuracy_df = pd.DataFrame(test_accuracy_data, index=["Test Accuracy"])
    test_accuracy_df = test_accuracy_df.transpose()
    
    test_roc_df = pd.DataFrame(test_roc_data)
    test_roc_df = test_roc_df.transpose()

    with pd.ExcelWriter(os.path.join(data_path, "analysis_results.xlsx")) as writer:
        validation_accuracy_df.to_excel(writer, sheet_name="Validation Accuracy")
        validation_roc_df.to_excel(writer, sheet_name="Validation AUC")
        test_accuracy_df.to_excel(writer, sheet_name="Test Accuracy")
        test_roc_df.to_excel(writer, sheet_name="Test AUC")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and analyze results.")
    parser.add_argument("--draw_plots", action="store_true", help="Draw plots if set.")
    args = parser.parse_args()

    data_path = "../data/xsens/models/results"
    collect_results(data_path, args.draw_plots)
