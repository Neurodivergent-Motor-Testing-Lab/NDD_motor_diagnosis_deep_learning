import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

title_size = 20
label_size = 15
tick_size = 12


def get_roc_metrics(experiment_path):
    roc_data = np.load(
        os.path.join(experiment_path, "all_auc.npy"), allow_pickle=True
    ).item()
    roc_means = {}
    roc_stds = {}
    for ndd in roc_data:
        roc_means[ndd] = np.mean(roc_data[ndd])
        roc_stds[ndd] = np.std(roc_data[ndd])
    return roc_data.keys(), roc_means, roc_stds


def get_accuracy_metrics(experiment_path):
    test_accuracy = np.load(os.path.join(experiment_path, "test_accuracy.npy"))
    mean_accuracy = np.mean(test_accuracy[-1])
    std_accuracy = np.std(test_accuracy[-1])
    return mean_accuracy, std_accuracy


def plot_single_metric(ax, metric, evaluation_timestep, title, xlabel, ylabel):
    num_epochs = metric.shape[0]
    if num_epochs == 0:
        return
    x_range = np.arange(num_epochs) * evaluation_timestep
    ax.clear()

    mean_metric = np.mean(metric, axis=1)
    ax.plot(x_range, mean_metric, label="Mean")
    std_metric = 2 * np.std(metric, axis=1)
    ax.fill_between(
        x_range, mean_metric - std_metric, mean_metric + std_metric, alpha=0.5
    )

    # ax.legend()
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_size)


def redraw_training_data(experiment_path, output_folder):
    evaluation_timestep = 10
    train_loss = np.load(os.path.join(experiment_path, "train_loss.npy"))
    test_loss = np.load(os.path.join(experiment_path, "test_loss.npy"))

    train_accuracy = np.load(os.path.join(experiment_path, "train_accuracy.npy"))
    test_accuracy = np.load(os.path.join(experiment_path, "test_accuracy.npy"))

    training_fig = plt.figure()
    ax_train_loss = training_fig.add_subplot(221)

    ax_test_loss = training_fig.add_subplot(222, sharex=ax_train_loss)

    ax_train_accuracy = training_fig.add_subplot(223, sharex=ax_train_loss)

    ax_test_accuracy = training_fig.add_subplot(
        224, sharex=ax_train_loss, sharey=ax_train_accuracy
    )

    plot_single_metric(
        ax_train_loss,
        train_loss,
        evaluation_timestep,
        "Training Loss",
        "# of Epochs",
        "Loss",
    )

    plot_single_metric(
        ax_test_loss, test_loss, evaluation_timestep, "Test Loss", "# of Epochs", "Loss"
    )

    plot_single_metric(
        ax_train_accuracy,
        train_accuracy,
        evaluation_timestep,
        "Training Accuracy",
        "# of Epochs",
        "Accuracy",
    )

    plot_single_metric(
        ax_test_accuracy,
        test_accuracy,
        evaluation_timestep,
        "Test Accuracy",
        "# of Epochs",
        "Accuracy",
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


def redraw_all_roc(experiment_path, output_folder, condition_map):
    num_roc_points = 500
    fpr = np.linspace(0, 1, num_roc_points)
    tpr = np.load(
        os.path.join(experiment_path, "all_tpr.npy"), allow_pickle=True
    ).item()
    colors = ["blue", "red", "green", "black"]

    without_std_fig = plt.figure()
    without_std_ax = without_std_fig.gca()

    with_std_fig = plt.figure()
    with_std_ax = with_std_fig.gca()

    for i, ndd in enumerate(tpr.keys()):
        ndd_tpr = tpr[ndd]
        ndd = condition_map[ndd]
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
        with_std_ax.fill_between(fpr, mean_tpr + std_tpr, mean_tpr - std_tpr, alpha=0.5)

    for fig, ax in [(with_std_fig, with_std_ax), (without_std_fig, without_std_ax)]:
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
    with_std_fig.savefig(
        os.path.join(output_folder, "all_roc_with_std.svg"), bbox_inches="tight"
    )
    plt.close(with_std_fig)

    without_std_fig.savefig(
        os.path.join(output_folder, "all_roc_without_std.svg"), bbox_inches="tight"
    )
    plt.close(without_std_fig)


def redraw_individual_roc(experiment_path, output_folder, condition_map):
    num_roc_points = 500
    fpr = np.linspace(0, 1, num_roc_points)
    tpr = np.load(
        os.path.join(experiment_path, "all_tpr.npy"), allow_pickle=True
    ).item()

    for ndd in tpr.keys():
        ndd_tpr = tpr[ndd]
        ndd = condition_map[ndd]
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


def redraw_figures(experiment_path, output_folder, condition_map):
    plt.ioff()
    redraw_training_data(experiment_path, output_folder)
    redraw_all_roc(experiment_path, output_folder, condition_map)
    redraw_individual_roc(experiment_path, output_folder, condition_map)
    return


def collect_results(data_path):
    MEAN_KEY = "Mean"
    STD_KEY = "Stdev"
    fold_map = {"66-33": "3 Fold", "75-25": "4 Fold", "80-20": "5 Fold"}
    condition_map = {"ASD": "ASD", "ADHD": "ADHD", "ASD+ADHD": "ASD+ADHD", "C": "NT"}
    signal_map = {
        "Linear Accel": "Linear Accel",
        "RPY_dot": "RPY_dot",
        "RPY": "RPY",
        "linear_accel_and_RPY_dot": "Linear Accel + RPY Dot",
        "linear_accel_and_RPY": "Linear Accel + RPY",
        "RPY_dot_and_RPY": "RPY dot + RPY",
        "all": "All",
    }
    output_figure_path = os.path.join(data_path, "figures")
    if not os.path.exists(output_figure_path):
        os.makedirs(output_figure_path)
    accuracy_data = {}
    roc_data = {}
    for signal in os.listdir(data_path):
        signal_path = os.path.join(data_path, signal)
        if os.path.isdir(signal_path) == False:
            continue
        if signal_path == output_figure_path:
            continue
        print(signal)
        signal_key = signal_map[signal]
        accuracy_data[signal_key] = {}
        roc_data[signal_key] = {}
        for fold in fold_map:
            accuracy_data[signal_key][(fold_map[fold], MEAN_KEY)] = None
            accuracy_data[signal_key][(fold_map[fold], STD_KEY)] = None
            for ndd in condition_map:
                ndd_key = condition_map[ndd]
                roc_data[signal_key][(fold_map[fold], ndd_key, MEAN_KEY)] = None
                roc_data[signal_key][(fold_map[fold], ndd_key, STD_KEY)] = None

        for train_ratio in os.listdir(signal_path):
            print("\t", train_ratio, fold_map[train_ratio])
            experiment_path = os.path.join(signal_path, train_ratio)
            mean, std = get_accuracy_metrics(experiment_path)
            fold_key = fold_map[train_ratio]
            accuracy_data[signal_key][(fold_key, MEAN_KEY)] = mean
            accuracy_data[signal_key][(fold_key, STD_KEY)] = std

            ndds, roc_means, roc_stds = get_roc_metrics(experiment_path)
            for ndd in ndds:
                ndd_key = condition_map[ndd]
                roc_data[signal_key][(fold_key, ndd_key, MEAN_KEY)] = roc_means[ndd]
                roc_data[signal_key][(fold_key, ndd_key, STD_KEY)] = roc_stds[ndd]
            figure_path = os.path.join(output_figure_path, signal, train_ratio)
            if not os.path.exists(figure_path):
                os.makedirs(figure_path)
            redraw_figures(experiment_path, figure_path, condition_map)

    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_df = accuracy_df.transpose()
    print(accuracy_df)

    roc_df = pd.DataFrame(roc_data)
    roc_df = roc_df.transpose()
    print(roc_df)
    with pd.ExcelWriter(os.path.join(data_path, "analysis_results.xlsx")) as writer:
        accuracy_df.to_excel(writer, sheet_name="Test Accuracy")
        roc_df.to_excel(writer, sheet_name="AUC")


if __name__ == "__main__":
    data_path = "../data/xsens/models/results"
    collect_results(data_path)
