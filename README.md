## 1. Environment construction

Install anaconda and run environment.yaml to build the virtual environment:

```shell
conda env create -f environment.yaml
` ` `

Or use the pip command to install:

```shell
pip install -r requirements.txt
` ` `

Wait for the dependency download to complete, activate the virtual environment:

```shell
conda activate camloc
` ` `

## 2. Data set

The data sets are placed in the same level folder as the project code.


## 3. Data parameter configuration

All parameters of the project are configured in '... /CamLoc/tools/options.py '. Read the parameters before running the code

## 4. Data mean and variance

Before model training, it is necessary to normalize the data and calculate the mean and variance. Change the configuration in options.py, specify the dataset and scenario, run 'dataset_mean.py', and a 'stats.txt' file will be generated under the corresponding scenario dataset.

```shell
python dataset_mean.py
` ` `

## 5. Test (Prediction)

Also change the configuration in options.py, specifying the data set and scene, batchsize, number of Gpus, number of epochs, and so on.

Indicates the trained model weights, for example: './logs/7Scenes_office_CAPLoc_False/models/epoch_100.pth.tar '

```shell
python eval.py
` ` `

## 6. Training

Also change the configuration in options.py, specifying the data set and scene, batchsize, number of Gpus, number of epochs, and so on

```shell
python train.py
` ` `

## 7. Significant visualization

Also change the configuration in options.py, specifying the data set and scene, batchsize, number of Gpus, number of epochs, and so on

** Most important ** : Indicates the trained model weights, such as: './logs/7Scenes_office_CAPLoc_False/models/epoch_100.pth.tar '

```shell
python saliency_map.py
` ` `

Generated visualization file in '... /logs/7Scenes_chess_CAPLoc_False/figures/7Scenes_chess_attention_CAPLoc.avi`