A prerequisite to the deep learning is to run the matlab code for each subject.
Then, preprocess the trial wise data using: 
```
cd src/xsens
python getTrialData.py
``` 
The above step only needs to be done once unless new participants are analyzed using the matlab code

After that, to train a DL model run 
```
cd src
python -m ML.train_lstm
```
