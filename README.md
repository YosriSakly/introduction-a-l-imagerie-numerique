# Image fusion using guided filtering

## Dependencies
We make extensive use of the library **opencv** for python 
and its extended version **opencv-contrib** which include the guided filtering code.
Both libraries can be installed using pip: 

```
$pip install opencv-python
$pip install opencv-contrib-python
```
We alsos use the the library **Pillow** for gif support. This is optionnal if one does not want to use gif input format. Else it can be installed using pip:  

```
$pip install Pillow
```

## Main functionnalities  
***Supported formats are jpg, png and gif***  

### 1/ Image fusion using guided filtering
In **main.py**, just change the **path** variable to the folder with the images to fuse and then run the script.  
*This method works for any number of possibly colored images*

### 2/ Image fusion with Laplacian pyramids
In **main_laplacian.py** just change the **path** variable to the folder with the images to fuse and then run the script.  
*This method works for two images in grayscale*  

