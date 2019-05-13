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

##Step 2: Testing of virtual data(Simulated data).
###2a.Installing requirements
[Anaconda](https://www.anaconda.com/distribution/). Download the anaconda along with desired python version. In this project, we will be using windows, conda version 4.6, python 3.7.  
After installing anaconda, open Anaconda Prompt as Administrator and create a new virtual environment by copying following commend. 

```
C:\> conda create -n env  pip python=3.6

```



