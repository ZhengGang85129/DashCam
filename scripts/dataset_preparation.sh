python3 ./src/data_preparation/dataset_split.py
cp ./dataset/test.csv ./dataset/img_database/test.csv
echo "Check meta data: dataset/img_database/test.csv"
#Prepare dataset
python3 ./src/data_preparation/prepare_image_database.py --metadata_path ./dataset/img_database/validation.csv --dataset validation --extract_frame

python3 ./src/data_preparation/prepare_image_database.py --metadata_path ./dataset/img_database/test.csv --dataset test --extract_frame

python3 ./src/data_preparation/prepare_image_database.py --metadata_path ./dataset/img_database/train.csv --dataset train --extract_frame