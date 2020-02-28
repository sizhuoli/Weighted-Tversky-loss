Separation challenge of the clumping objects in the image segmentation tasks

Here I propose a method of using weighted Tversky loss to solve the separation challenge for the task of predicting the number of objects from images via U-Net.

The method involves the computation of the weight maps from ground truth labels, integration of the weight maps with the traditional U-Net framework, and the implemtation of the weighted Tversky loss.

Two different ways of creating the weight maps are proposed. 
