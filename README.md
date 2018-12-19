# CS-433 - Road Segmentation
Project in CS-433 - Machine Learning, EPFL (Fall 2018).

Group: Maiken Berthelsen, Ida Sandsbraaten and Sigrid Wanvik Haugen.




### Running instructions:
- Packages that need to be installed are:
	- pip install ......
	- The Python version used is 3.6.5.
	- Keras version is 2.1.6-tf.

- Download the .zip-file containing the data set from https://www.crowdai.org/challenges/epfl-ml-road-segmentation/dataset_files. Make sure that the folder 'data' containing these files are in the same folder as the src folder.

- The best results obtained at crowdAI, can be reproduced by calling "python run.py" from your terminal. Make sure that you are in the src folder when running. This will load the best model and produce the submission file based on the implemented model and weights.

- The running is finished when "Finished" is printed to the terminal.

- The submission csv file can be found in the same folder with the name "submission.csv".

- If you want to train the model, instead of loading the already trained one, change ____ .




### Code architecture:
This code contains the following files in the src folder:

* run.py 

	- The reults will be reproduced when running run.py.
	
	
* unet.ipynb (Jupyter Notebook)

	- The notebook for training our U-net model.


* cnn.ipynb (Jupyter Notebook)

	The notebook for training our basic CNN model.
	
	
* unet_model.py

	Contains the U-net model.


* cnn_model.py

	Contains the basic CNN model.


* data_extraction.py


* image_processing.py


* image_augmentation.py
	
	
* mask_to_submission.py

	Code provided to us.
	




#### Misc


numpy

Scipi

Tensorflow
tensorflow

Keras
pip install Keras

Correct  error
pip install h5py==2.8.0rc1 
