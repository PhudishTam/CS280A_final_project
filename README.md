# CS280A_final_project

### Set Up
We have provided the environment.yml file so you can install the environment by simply run conda env create -f environment.yml 

### Running Experiments
Once we have install the environment we can start to run for a few experiments. You can download the dataset that we use from this website 
"https://cocodataset.org/#download" note that our dataset is the combination of the 2015 one with the 2017 one. Once the dataset is downloaded then 
make sure that you set the path into the src/hparams/GAN.json for the train_data_dir and test_data_dir. Once that is done you can simply either run 
python src/model/train_GAN.py to run the training for the first experiment, python src/model/train_pre_train.py to run the second experiment and python src/model/train_VQ_GAN.py to run the third experiment. Note that you should also change your save directory so that we can use it later for inference.

### Running Inference
Once the trainning is done we can do some inference in the following file python src/model/inference_GAN.py for first experiment, python src/model/inference_pretrain_GAN.py for second experiment and python src/model/inference_VQGAN.py for the third experiment. These files will save the image and also the training and testing losses as well.