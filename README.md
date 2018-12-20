# CS-433 - Road Segmentation
Project in CS-433 - Machine Learning, EPFL (Fall 2018).

Group: Maiken Berthelsen, Ida Sandsbraaten and Sigrid Wanvik Haugen.

Maximum achieved F1 score: 0.869.



### Running instructions:

- The code runs on Python version 3.6.5, with Keras version 2.1.6-tf with TensorFlow backend. In order to install all the packages required for this code, make sure you have the correct version of Python (3.6.5), and **pip3** installed on your computer. Then copy paste

	```
	pip3 install -r requirements.txt
	```
	in your terminal when in the main folder, and press enter.

- Download the .zip-file containing the data set from [HERE](https://www.crowdai.org/challenges/epfl-ml-road-segmentation/dataset_files). Make sure that the folder 'data' containing these files are in the same folder as the src folder, as in the File structure shown below.

- The best results obtained at crowdAI, can be reproduced by calling 
	```
	python3 src/run.py
	```
	from your terminal, when in the directory of the main folder. This will load the best model and produce the submission file based on the implemented model and weights. The running is finished when "Finished." is printed to the terminal. The submission csv file can be found in the same folder with the name "submission.csv".

- If you want to train the models from scratch, instead of loading the already trained one, you can use the Jupyter Notebooks CNN.ipynb (for the CNN model) and U_Net.ipynb (for the U-Net model). The Jupyter Notebooks can be used via [Anaconda](https://www.anaconda.com/).

- We obtained the results by running the notebooks in [Google's Colaboratory](https://colab.research.google.com/) and using their GPU. The files found in the src must then be uploaded to the Colaboratory. In order to avoid timed-out error by running several epochs, the model.fit()-command found under **Training the model** can be copied into several blocks, and then running only 5 or 10 epochs in each block.




### File structure:
The main folder should have the following structure:

```
-- main_folder/
	- data/
		- test_set_images/
		- training/

	- src/
		- run.py 
		- U_Net.ipynb 
		- CNN.ipynb
		- unet_model.py
		- cnn_model.py 
		- data_extraction.py
		- image_processing.py
		- image_augmentation.py
		- mask_to_submission.py

	- weights/
		- UNET_best_weights.hdf5
	
```

In the data folder, all the images used to train and test the model are stored. The src folder contains all the code, better described below. The weights folder contains the weigths for the U-Net model which reproduces the best result on CrowdAI, which is done by running 

```
Python3 src/run.py
```
as described above.



### Overwiev of Python files

* run.py 

	- This script reproduces the best result.
	
* U_Net.ipynb (Jupyter Notebook)

	- The notebook for training the U-Net model.


* CNN.ipynb (Jupyter Notebook)

	- The notebook for training the CNN model.
	
	
* unet_model.py

	- Contains the U-net model.


* cnn_model.py

	- Contains the CNN model.


* data_extraction.py


* image_processing.py


* image_augmentation.py
	
	
* mask_to_submission.py

	

### Credits

Much of the source code is based on the code given to us from the course, written by Aurelien Lucchi, ETH Zürich, found [HERE](https://github.com/epfml/ML_course/blob/master/projects/project2/project_road_segmentation/tf_aerial_images.py). The U-Net architechture is based on Tobias Sterkbak's, found [HERE](https://www.depends-on-the-definition.com/unet-keras-segmenting-images/).
