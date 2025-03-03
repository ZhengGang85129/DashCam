# DashCam
## Environment setting:

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
```

## Dataset Download and Offline-preprocessing(must)
```
kaggle competitions download -c nexar-collision-prediction
```
* This requires your authentication key and account ID, please follow [here](https://github.com/Kaggle/kaggle-api#download-dataset-files) for more details.

* After downloading the data and unfolding the zip file, please rename the folder as "dataset". Then, run "./traindataset_frames_extraction.py" to do the data preprocessing.
```
python3 ./traindataset_frames_extraction.py
```
## Dataset splitting(must)
Use following step to split the dataset(train) into train and validation. To keep consistency over our team, do not modify anything in the script.
```
pip3 install scikit-learn
python3 ./dataset_split.py
``` 


## Training (To-Be-Done)
Simple command-line example:
```
python3 train.py --batch_size ${batch_size} --learning_rate ${learning_rate} --monitor_dir ${monitor_dir} 
```

## Download DAD dataset
```
pip3 install gdown;
gdown 1Z_vUmhGe4lES0ASUcROK58q7fjFTU_Y3;
```