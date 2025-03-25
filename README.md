# DashCam
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
```

## Step 1. Dataset Download
```
kaggle competitions download -c nexar-collision-prediction;
unzip nexar-collision-prediction.zip; # 29.2 G, it may take up to 20 minutes.
mv 
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
### Step 2.1 Dataset splitting (Dev)
Follow these steps to partition the training dataset into training and validation subsets. To maintain consistency across our team, please execute this script without any modifications.
```
python3 ./utils/dataset_split.py
```
Once you execute the above command, you will see the new csv files:
```
./dataset
    - ...
    - sliding_window/
        train_video.csv
        validation_video.csv
    - ...
```


### Step 2.2 Dataset preprocessing for training (Dev)
With following command, you will have the preprocessed videos for sliding-window approach(dev).
```
python3 ./utils/prepare_training_dataset.py --dataset train # size: 1200 by default
python3 ./utils/prepare_training_dataset.py --dataset validation # size: 300 by default
```
You will see extracted clips as:
```
./dataset
    - ...
    - sliding_window
        train_extracted_clips.csv # columns: (id,start_frame,end_frame,event_frame,alert_frame,target) 
        validation_extracted_clips.cvs # columns: (id,start_frame,end_frame,event_frame,alert_frame,target)
        - train
            0*.mp4
            ...
        - validation
            0*.mp4
            ...
```
### Step 2.3 Dataset preprocessing for evaluation
This preprocessing will use the validation videos to create clips sequence of contiguous frames with last frame being the accident frame.
```
python3 ./utils/prepare_validation_dataset.py 
```
Structure:
```
- dataset
    - ...
    - sliding_window
        - evaluation
            - tta_500ms
                0*.mp4
                ...
            - tta_1000ms
                0*.mp4
                ...
            - tta_1500ms
                0*.mp4
        ...
    ...
```

## Training (To-Be-Done)
Simple command-line example:
```
python3 train.py --batch_size <batch_size> --learning_rate <learning_rate> --monitor_dir <monitor_dir> 
```

## Analysis Visualization 
### Probability distribution across frame indices
```
python3 ./accident_prediction_vis.py --clip_path dataset/train/train_video/00043.mp4 --model_ckpt CHECKPOINT --filename probability --model_type [baseline:timesformer:swintransformer]
```
### Likelihood & Recall vs Precision curve:
```
python3 ./score_assessment.py --model_ckpt <path_to_modelckpt> --num_workers 8
```

## Inference
```
python3 ./submission.py --num_workers 8 --model_ckpt <path_to_your_ckpt> --model_type baseline
kaggle competitions submit -c nexar-collision-prediction -f submission.csv -m "Message"
```