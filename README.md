Python 3.7.6

steps:
python -m venv env
env\Scripts\Activate
pip install tensorflow==1.15
pip install -r ./TecoGAN-master/requirements.txt
pip install torch==0.4.1 -f https://download.pytorch.org/whl/torch_stable.html

cd TecoGAN-master
python runGan.py 0
python runGan.py 1