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

## Dataset Download and Offline-preprocessing
```
kaggle competitions download -c nexar-collision-prediction
```
* This requires your authentication key and account ID, please follow [here](https://github.com/Kaggle/kaggle-api#download-dataset-files) for more details.

* After downloading the data and unfolding the zip file, please rename the folder as "dataset". Then, run "./traindataset_frames_extraction.py" to do the data preprocessing.
```
python3 ./traindataset_frames_extraction.py
```

## Training (To-Be-Done)
```
python3 ./train.py
```

## Download DAD dataset
```
pip3 install gdown;
gdown 1Z_vUmhGe4lES0ASUcROK58q7fjFTU_Y3;
```