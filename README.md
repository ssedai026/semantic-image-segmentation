Semantic segmentation training for images based on tensorflow keras. It
supports segmentation of multiple classes. it provides easy interface
for training. Just put images and labels in the given format in a
folder.

Steps:
* Prepare the data in a given format 
  *   Label maps should be of a gray scale image where each pixel
      contains value of the class label. If the segmentaion problem have
      C classes then the value of the label map pixel can range from [0
      to C -1)
  * Example mnist data for multiclass segmentation can be generated
    using mnist_generate_data.py
    
*  Define a configuration class for your experiment and put the training
   images and label map path. Define your experiment name, log path and
   other trainining parameters.  

* Pass the configuration to the training function. Example training
  script : train_mnist.py
  
* All the training results will be saved in
  config.LOG_ROOT_DIR/config.NAME directory. This included visualization
  of segmentation of each epoch and losses.
 

This code has been tested on tensorflow 1.12.0 and python 3.6.x. There
is an issue on tensorflow version > and < 2.0.2
   
  
  

Example segmentation training on toy mnist dataset is provided in
train_mnist.py
  
 ![Example output](.github/res/mnist_result.png)
