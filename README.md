# This repository is for Team Project Deep learning with simulation

In order to access all the files clone this repository.

## There are three main steps in this project. 
step 1: Generation of simulated data (Virtual Data).  
step 2: Testing of virtual data.  
step 3: Using Inception v2 Algorithm.  

## Step 1 : Generation of simulated data(virtual data)
### Prerequisites 
[Blender - Install blender by clicking this](https://www.blender.org/download/).  
Clone this repository you have all the files needed for this step under the folder Python_code_for_rendering and blender_files.  
### Procedure:
Run the blender application, change the mode to scripting form the above panel on the top. Then import the file generate more data panel and execute it.  
If there are any problems in running the script we made a blend file in which the generate more data panel is preloaded. Just open the file ‘generateMoreData.blende’. Navigate to file -> append and Navigate to blender_files in the explorer and select the model(which is in the format of .blend file), you get to see more options navigate to object and select all the files and click ‘Append from library’ then you get your model imported in blender. Run the script by clicking ‘Reload Trusted’  you need to see something like in the image.  
folder
  
<p align="center">
  <img src="Images/1.JPG" width="750" title="Loaded model in blender">
</p>
  
You can play with the settings form the GenerateMoreData pannel(I know, I am bad at giving names) after your desired settings just click ‘Animation’ form the panel to the right side.(Do not forget to change the resolution and folder path to save the image files code automatically creates a floder in the path).  

## Step 2: Testing of virtual data(Simulated data).
### 2a.Installing requirements
Download the [Anaconda](https://www.anaconda.com/distribution/) along with desired python version. In this project, we will be using windows, conda version 4.6, python 3.6.  
After installing anaconda, open Anaconda Prompt as Administrator and create a new virtual environment by copying following command. 

```
conda create -n env  pip python=3.6
```
Then activate the newely created enviroment.  
```
(base) C:\Windows\system32>conda activate env
(env) C:\Windows\system32>
```

Install tensorflow in this environment by issuing:  

```
(env) C:\Windows\system32> pip install --ignore-installed --upgrade tensorflow
```
If you want to use GPU version of the tensorflow you can follow instructions in this [youtube video](https://www.youtube.com/watch?v=59duINoc8GM). 
Install keras and other packages  
```
(tensorflow1) C:\> conda install -c anaconda keras
(tensorflow1) C:\> conda install -c anaconda keras-gpu // skip this if you want to use only CPU version
(tensorflow1) C:\> conda install spyder
(tensorflow1) C:\> conda install -c anaconda scikit-learn
```
Install the other necessary packages by issuing the following commands:
```

(tensorflow1) C:\> pip install lxml
(tensorflow1) C:\> pip install Cython
(tensorflow1) C:\> pip install matplotlib
(tensorflow1) C:\> pip install pandas
(tensorflow1) C:\> pip install opencv-python
```

