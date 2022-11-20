curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -o Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh




aws s3 cp --recursive s3://ejkquant-uswest1/train_data ./train_data
pip install pandas pyarrow matplotlib click
git clone https://github.com/eskui/Cost_Optimal_ML_Workflow.git
python3 train_on_batch.py 2020 7200 5 NVIDIA_T4
python3 train_on_batch.py 2020 7200 6 local_CPU
