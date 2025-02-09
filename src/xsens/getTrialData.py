import scipy.io as sio
import numpy as np
import os
import argparse
import fnmatch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create Trial wise data.")
    parser.add_argument("--input_folder", help="Path to Matlab data")
    parser.add_argument("--output_folder", help="Path to export the python data")
    args = parser.parse_args()
    base_folder = "../../../NDD_motor_diagnosis_subject_data_analyzed"

    if args.input_folder != None:
        base_folder = args.input_folder

    op_path = "../../data/xsens/data/"
    if args.output_folder != None:
        op_path = args.output_folder
    all_data = {}
    all_data["NT"] = []
    all_data["ASD"] = []
    all_data["ADHD"] = []
    all_data["A^2"] = []
    for subject_folder in os.listdir(base_folder):
        if os.path.isdir(os.path.join(base_folder, subject_folder)) == False:
            continue
        subject_diagnoses_file = os.path.join(
            base_folder, subject_folder, "diagnosis.txt"
        )
        with open(subject_diagnoses_file, "r") as f:
            subject_diagnosis = f.readline().strip()
        subject_mat_file = os.path.join(
            base_folder, subject_folder, "trial_wise_data.mat"
        )
        if os.path.isfile(subject_mat_file):
            subject_data_mat = sio.loadmat(subject_mat_file)
            subject_trial_data = np.array(subject_data_mat["trial_data"])
            print(subject_trial_data.shape)
            # subject_trial_list = subject_trial_data.tolist()
            subject_trial_list = list(subject_trial_data)

            all_data[subject_diagnosis].extend(subject_trial_list)
        else:
            print("ERROR", subject_folder, " does not have sensor_data")

    print(len(all_data["NT"]))
    print(len(all_data["ASD"]))
    print(len(all_data["ADHD"]))
    print(len(all_data["A^2"]))

    # print(all_data)
    save_path = os.path.join(op_path, "all_trial_data")
    np.save(save_path, all_data)