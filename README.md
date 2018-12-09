# CS-433 - Road Segmentation
Project in CS-433 - Machine Learning, EPFL (Fall 2018).

Group: Maiken Berthelsen, Ida Sandsbraaten and Sigrid Wanvik Haugen.




### Running instructions:
- Packages that need to be installed are:
	- pip install ......
	- The Python version used is 3.6.5.
	- Keras version is 2.1.6-tf.

- Download the .zip-file containing the data set from https://www.crowdai.org/challenges/epfl-ml-road-segmentation/dataset_files. Make sure that the folder 'data' containing these files are in the same folder as the src folder.

- The code can be run by calling "python run.py" from your terminal. Make sure that you are in the src folder when running. This will load the best model and produce the submission file based on the implemented model and weights.

- The running is finished when "Finished" is printed to the terminal.

- The submission csv file can be found in the same folder with the name "results.csv".

- If you want to train the model, instead of loading the already trained one, change ____ .




### Code architecture:
This code contains the following files:
* run.py 

	The whole code can be executed by running run.py. This file imports helpers.py and implementations.py

* helpers.py

	Contains the given functions for loading csv, predicting labels and create a csv submission.

* implementations.py

	Contains all of our implementations and is divided into three parts, each containing:

		- The six ML methods that are asked for in the project description.

		- The functions for data processing.

		- Contains the functions used in the ML methods.

* validation.py

	Cross-validation that is used for the different methods is implemented here.

* run_functions.py
	
	The functions for actually running the implementation is placed in this file as well as the tuned parameters for the different functions.

















## Misc


numpy

Scipi

Tensorflow
tensorflow

Keras
pip install Keras

Correct  error
pip install h5py==2.8.0rc1 
