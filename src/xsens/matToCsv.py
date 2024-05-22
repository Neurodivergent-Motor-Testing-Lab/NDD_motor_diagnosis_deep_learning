import scipy.io as sio
import numpy as np
import os
import argparse
import fnmatch
import shutil

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create Trial wise data.")
    parser.add_argument("--input_folder", help="Path to Matlab data")
    parser.add_argument("--output_folder", help="Path to export the python data")
    args = parser.parse_args()
    base_folder = "../../../Data"

    if args.input_folder != None:
        base_folder = args.input_folder

    op_path = "../../../Data_clean"
    if args.output_folder != None:
        op_path = args.output_folder
    for subject_folder in os.listdir(base_folder):
        if os.path.isdir(os.path.join(base_folder, subject_folder)) == False:
            continue
        if not os.path.exists(os.path.join(op_path, subject_folder)):
            os.makedirs(os.path.join(op_path, subject_folder))

        subject_mat_file = os.path.join(base_folder, subject_folder, "sensor_data.mat")
        if os.path.isfile(subject_mat_file):
            subject_data = sio.loadmat(subject_mat_file)
            person_name = [x for x in subject_data.keys() if "__" not in x]
            csv_data = np.array(subject_data[person_name[0]])
            csv_path = os.path.join(op_path, subject_folder, "sensor_data.csv")
            np.savetxt(csv_path, csv_data, delimiter=",")

        else:
            print("ERROR", subject_folder, " does not have sensor_data")
        subject_diag_file = os.path.join(base_folder, subject_folder, "diagnosis.txt")

        if os.path.isfile(subject_diag_file):
            shutil.copyfile(
                subject_diag_file,
                os.path.join(op_path, subject_folder, "diagnosis.txt"),
            )
        else:
            print("ERROR", subject_folder, " does not have a diagnosis")
