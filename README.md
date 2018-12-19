# CS-433 - Road Segmentation
Project in CS-433 - Machine Learning, EPFL (Fall 2018).

Group: Maiken Berthelsen, Ida Sandsbraaten and Sigrid Wanvik Haugen.




### Running instructions:

The code was ran on Python version 3.6.5., with Keras version 2.1.6-tf with TensorFlow backend. In order to install all the packages required for this code, is to run

```
pip3 install -r requirements.txt
```


- Download the .zip-file containing the data set from https://www.crowdai.org/challenges/epfl-ml-road-segmentation/dataset_files. Make sure that the folder 'data' containing these files are in the same folder as the src folder.

- The best results obtained at crowdAI, can be reproduced by calling "python3 src/run.py" from your terminal, when in the directory of the main folder. Make sure that you are in the src folder when running. This will load the best model and produce the submission file based on the implemented model and weights.

- The running is finished when "Finished." is printed to the terminal.

- The submission csv file can be found in the same folder with the name "submission.csv".

- If you want to train the models from scratch, instead of loading the already trained one, you can use the Jupyter Notebooks CNN.ipynb (for the CNN model) and U_Net.ipynb (for the U-Net model).




### Code architecture:
This code contains the following files in the src folder:

* run.py 

	- The reults will be reproduced when running run.py.
	
	
* U_Net.ipynb (Jupyter Notebook)

	- The notebook for training our U-Net model.


* CNN.ipynb (Jupyter Notebook)

	- The notebook for training our basic CNN model.
	
	
* unet_model.py

	- Contains the U-net model.


* cnn_model.py

	- Contains the basic CNN model.


* data_extraction.py


* image_processing.py


* image_augmentation.py
	
	
* mask_to_submission.py

	- Code provided to us.
	




#### Misc



