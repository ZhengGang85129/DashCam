# DashCam
This repository contains our solution for the [Kaggle competition](https://www.kaggle.com/competitions/nexar-collision-prediction/overview), where was achieved 16th place (Top7%).

## Model Backbone:
Our original solution utilized the **R3D-18** backbone. After reviewing the top-5 solutions in the competition, we decided to improve our model by replacing the R3D-18 backbone with **Multi-Scale Vision Transformer v2 (MViT v2)**. This modification led to a significant performance boost, achieving a mean Averaged Precision(mAP) score of **0.843**, surpassing the performance of the 5th-place solution, which also showcases the effectiveness of model-centric improvement.

## Data Preprocessing:
* Training/Validation Split: We used 1200 training and 300 validation samples. In each epoch, we randomly sampled 16 consecutive frames per video.
* Exponential Cross-Entropy Loss: We implemented a custom exponential cross-entropy loss to emphasize increasing risk levels closer to an accident event. This approach helped the R3D-18 model achieve **~0.778** on the private leaderboard. However, after switching to MViT v2, we observed a performance drop. To mitigate the discrepancy between two consecutive frames, we conducted extensive hyperparameter tuning and found that a specific value of 1 provided the best solution.

## Data Augmentation:
For data augmentation, we used the following techniques to enhance the model's robustness and generalization:
* Color Jitter
* Random Rotation
* Horizontal Flip
  
## Training Details:
We employed an AdamW optimizer with a cosine annealing scheduler and early stopping after 10 epochs. The maximum number of epochs was set to 15, but we observed that the model converged between **9 and 11 epochs**.

## Hyperparameter Details:
We systematically performed hyperparameter optimization using Optuna iteratively, with the metric being the loss. The key hyperparameters tuned were:
* Learning rate
* Weight decay rate
* batch_size
* Stride size between two consecutive frames
* Classifier structures
  
# Training Time:
* The model was trainined on RTX-3090, with an average training time of **20 minutes**. 

## Step 0. Environment setting:

```
conda config --set solver classic
conda create -n dashcam python=3.11
conda activate dashcam
pip3 install kaggle
pip3 install pandas
pip3 install decord 
pip3 install torch 
pip3 install torchvision
pip3 install opencv-python
pip3 install av  
pip3 install matplotlib 
pip3 install torchsummary
pip3 install pyyaml
pip3 install timesformer-pytorch
pip3 install scikit-learn
pip3 install mlflow
pip3 install optuna
```

## Step 1. Dataset Download
```
kaggle competitions download -c nexar-collision-prediction;
unzip nexar-collision-prediction.zip -d dataset; # 29.2 G, it may take up to 20 minutes.
```
* This requires your authentication key and account ID, please follow [here](https://github.com/Kaggle/kaggle-api#download-dataset-files) for more details.

Once you unzip the zipped file, please rename the folder as "dataset". And You will see the dataset structured as:
```
./dataset
    - train # directory containing raw training videos with duration ~40 sec
    - test # directory containing raw test videos with duration ~10 sec
    train.csv # information of training videos. columns: (id, time_of_event, time_of_alert, target)
    test.csv # information of test videos. columns: (id)
    sample_submission.csv # sample format for submission.csv
```

## Step 2. Dataset Preprocessing
### Step 2.1 Dataset splitting and pre-processing 
Follow these steps to partition the training dataset into training and validation subsets. To maintain consistency across our team, please execute this script without any modifications.
```
sh ./scripts/dataset_preparation.sh
```
Once you execute the above command, you will see the new csv files:
```
./dataset
    - ...
    - img_database/
        frame-metadata_validation.csv
        frame-metadata_test.csv
        frame-metadata_train.csv
        train.csv
        test.csv
        - train #size: 1200
            video_XXXXX/
            ...
        - validation #size: 300
            video_XXXXX/
            ...
        - test/
            video_XXXXX/
            ...
    - ...   
```


## Step 3. Training
### 3.1 Simple command-line example:
```
python3 -m scripts.train configs/mvit2.yaml
```
### 3.2 For hyperparameter optimization:
To obtain the best hyperparameters, we have implemented `optuna` lib with n_trials = 50. 
```
sh ./scripts/run_experiments.sh
```


## Step 4.Visualization 

<img src="example/negative.gif" alt="Description of the image" width="500"/>

<img src="example/positive.gif" alt="Description of the image" width="500"/>
One can use the following command to view the risk level over time (frames).

```
CONFIG_YAML=configs/mvit_v2.yaml VIDEO_ID=XXX python3 -m visualization.visualize_risk_level 
```
