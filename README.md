# Image Segmentation Parallelisation

This project was for my HPC course, where we aimed to exploit the highly parallelizable aspect of image processing by parallellizing several image segmentation algorithms. This was done using CUDA, OpenMP and MPI, and the algorithms were benchmarked against their respective serial implementations.

The algorithms used for testing were:  
* K-Means Clustering algorithm
* Nick Thresholding algorithm
* Canny Edge Detection algorithm

[Results](https://github.com/ZubairBul2ia/Image-Segmentation-Parallelisation/blob/main/Project_slides.pdf) showed that the K-Means clustering algorithm largely benefitted from the CUDA implementation for larger images, and for larger numbers of clusters. Nick Thresholding and Canny Edge detection benefitted for larger images, whereas for smaller ones the Serial and CPU-parallelised versions fared better due to the pipeline nature of these algorithms. 
