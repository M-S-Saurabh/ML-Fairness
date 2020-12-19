----------------------------
    Files Description
----------------------------
- Describes files which can be executed like 'python <file-name>'
- Please use the 'requirements.txt' file with pip to install necessary dependencies. 'pip install -r requirements.txt'

ThresholdClassifier.py: 
    - Runs a threshold classifier on COMPAS dataset 
    - generates ROC, group-wise TPR, FPR and Prob graphs

adversarial_debiasing.py:
    - Runs a 2-layer neural network without de-biasing and with de-biasing on COMPAS data
    - Uses tensorflow-1.15
    - prints results before and after de-biasing
    - generates comparison graphs

reweighing_preproc.py:
    - Run re-weighing on logsitic regression model
    - Displays and generates comparison grpahs 

