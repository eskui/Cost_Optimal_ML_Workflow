aws s3 cp --recursive s3://ejkquant-uswest1/train_data ./train_data
pip install pandas pyarrow matplotlib
python3 train_on_batch.py 2020 7200 5 NVIDIA_T4
