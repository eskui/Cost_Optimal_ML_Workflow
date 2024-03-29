curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -o Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh


git clone https://github.com/eskui/Cost_Optimal_ML_Workflow.git

aws s3 cp --recursive s3://ejkquant-uswest1/train_data ./train_data
pip install pandas pyarrow matplotlib click

python3 train_model.py 2017 1440 G5_A10GPU 16000
python3 train_model.py 2017 1440 G3S_M60GPU 16000
python3 train_model.py 2017 1440 C5D_CPU 16000
