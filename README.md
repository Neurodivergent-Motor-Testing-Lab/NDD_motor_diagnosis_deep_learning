1) A prerequisite to the deep learning is to run the matlab code for each subject.
Then, preprocess the trial wise data using: 
```
cd src/xsens
python getTrialData.py
``` 
The above step only needs to be done once unless new participants are analyzed using the matlab code

2) After that, to train a DL model with hyperparameter tuning run 
```
cd src
python -m DL.train_lstm all --enable_early_stopping
```

3) Once all signals have been analyzed, you can create tabular data about the final metrics. 
To do so, there must exist a folder, containing a subfolder for each signal. 
Inside that subfolder must lie all the contents of the DL experiment output. Then, run
```
cd src
python collect_statistics.py
```

4) To run the experiment with random swapping of labels, run
```
cd src
python -m DL.random_permutations all --enable_early_stopping
```

5) To compute plots for the SHAP values, run
```
cd src
python -m interpret.shap_values all --experiment_path <path_to_output_from_3>
``` 