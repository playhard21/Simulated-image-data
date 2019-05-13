# This repository is for maintaining structure for Team Project Deep learning with simulation

In order to access all the files clone this repository.

## There are three main steps in this project. 
step 1: Generation of simulated data (Virtual Data).  
step 2: Testing of virtual data.  
step 3: Using Inception v2 Algorithms.  

## Step 1 :
### Prerequisites 
[Blender - Install blender  form](https://www.blender.org/download/).  
In the cloned repository you have all the files needed for this step under the folder Python_code_for_rendering and blender_files.  
### Procedure:
Run the blender application, change the mode to scripting form the above panel on the top. Then import the file generate more data panel and execute it.  
If there are any problems in running the script we made a blend file in which the generate more data panel is preloaded. Just open the file ‘generateMoreData.blende’. Navigate to file -> append and Navigate to blender_files in the explorer and select the model(which is in the format of .blend file), you get to see more options navigate to object and select all the files and click ‘Append from library’ then you get your model imported in blender. Run the script by clicking ‘Reload Trusted’  you need to see something like in the image.  
folder
<p align="center">
  <img src="images/1.JPG">
</p>

You can play with the settings form the GenerateMoreData pannel(I know, I am bad at giving names) after your desired settings just click ‘Animation’ form the panel to the right side.(Do not forget to change the resolution and folder path to save the image files code automatically creates a floder in the path).  
