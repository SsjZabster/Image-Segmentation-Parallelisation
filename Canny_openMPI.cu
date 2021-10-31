#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "mpi.h"


// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include "helper_functions.h"    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include "helper_cuda.h"         // helper functions for CUDA error check

// Define the files that are to be save and the reference images for validation
const char *imageFilename = "NewsPaper1080.pgm";
//const char *refFilename   = "ref_threshold.pgm";


//function declarations
//void createfilt(char a, float** arr, int size);
void createfilt2(char a, float* arr, int size);
void convolve(float * output, float* input, float* filt, int height, int width, int k_size, char type);
void GetMag(float * output, float* imx, float* imy, int height, int width);
void GetDir(float * output, float* imx, float* imy, int height, int width);
void NMSuppress(float* output, float* G,float* after_Gx, float* after_Gy, float thr_min, float thr_max, int width, int height);
void Hysteresis(float* output, float* nms,float* after_Gx, float* after_Gy, float thr_min, float thr_max, int width, int height);

//main program
int main(int argc, char **argv)
{
  int comm_size, ID;

    float *img_gauss;
    float *img_gx;
    float *img_gy;
    float *img_gMag;
    float *img_gDir;
    float *img_nms;
    float *img_fOut;

    float *Simg_gauss;
    float *Simg_gx;
    float *Simg_gy;
    float *Simg_gMag;
    float *Simg_gDir;
    float *Simg_nms;
    float *Simg_fOut;

    float *hData = NULL;

   //timer for the total program run time

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &ID);

      StopWatchInterface *timer = NULL;
   sdkCreateTimer(&timer);
   sdkStartTimer(&timer);
  

   // allocate the filter array
   int k_size = 3; //same for rows and cols
   float *gauss = (float *) malloc(k_size * k_size * sizeof(float));
   float *sobel_x = (float *) malloc(k_size * k_size * sizeof(float));
   float *sobel_y = (float *) malloc(k_size * k_size * sizeof(float));
   //char f_typ = 's';

    // use to create filters
    createfilt2('s', gauss, k_size);
    createfilt2('x', sobel_x, k_size);
    createfilt2('y', sobel_y, k_size);

    // load image from disk
    unsigned int width, height;

    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if(ID == 0){


    
    

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);


     img_gauss = (float *) malloc(size);
     img_gx = (float *) malloc(size);
     img_gy = (float *) malloc(size);
     img_gMag = (float *) malloc(size);
     img_gDir = (float *) malloc(size);
     img_nms = (float *) malloc(size);
     img_fOut = (float *) malloc(size);


   }
   
   		
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  

  //printf("\n\n\n\n\n%i,  %i\n\n\n\n\n", width,height);

  float* ShData = (float*)malloc(sizeof(float)*width*(height/comm_size));
		
  MPI_Scatter(hData, width*(height/comm_size), MPI_FLOAT, 
				ShData, width*(height/comm_size), 
				MPI_FLOAT, 0, MPI_COMM_WORLD);

  

    //create output array
     Simg_gauss = (float*)malloc(sizeof(float)*width*(height/comm_size));
     Simg_gx = (float*)malloc(sizeof(float)*width*(height/comm_size));
     Simg_gy = (float*)malloc(sizeof(float)*width*(height/comm_size));
     Simg_gMag = (float*)malloc(sizeof(float)*width*(height/comm_size));
     Simg_gDir = (float*)malloc(sizeof(float)*width*(height/comm_size));
     Simg_nms = (float*)malloc(sizeof(float)*width*(height/comm_size));
     Simg_fOut = (float*)malloc(sizeof(float)*width*(height/comm_size));

  MPI_Scatter(img_gauss, width*(height/comm_size), MPI_FLOAT, 
				Simg_gauss, width*(height/comm_size), 
				MPI_FLOAT, 0, MPI_COMM_WORLD);
 
  MPI_Scatter(img_gx, width*(height/comm_size), MPI_FLOAT, 
				Simg_gx, width*(height/comm_size), 
				MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  MPI_Scatter(img_gy, width*(height/comm_size), MPI_FLOAT, 
				Simg_gy, width*(height/comm_size), 
				MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  MPI_Scatter(img_gMag, width*(height/comm_size), MPI_FLOAT, 
				Simg_gMag, width*(height/comm_size), 
				MPI_FLOAT, 0, MPI_COMM_WORLD);
 
  MPI_Scatter(img_gDir, width*(height/comm_size), MPI_FLOAT, 
				Simg_gDir, width*(height/comm_size), 
				MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  MPI_Scatter(img_nms, width*(height/comm_size), MPI_FLOAT, 
				Simg_nms, width*(height/comm_size), 
				MPI_FLOAT, 0, MPI_COMM_WORLD);
 
  MPI_Scatter(img_fOut, width*(height/comm_size), MPI_FLOAT, 
				Simg_fOut, width*(height/comm_size), 
				MPI_FLOAT, 0, MPI_COMM_WORLD);
   
  for(int z = 0; z < width;z++){
	for(int k = 0; k < height/comm_size;k++){

		//printf("%f",ShData[z*width+k]);

		}
	}

	 	  	    char k[1024];
    snprintf( k, 10, "%d", ID );

    strcpy(k + strlen(imagePath) - 4, "_gauss.pgm");
	sdkSavePGM(k, ShData, width, height);


   StopWatchInterface *timer1 = NULL;
   sdkCreateTimer(&timer1);
   sdkStartTimer(&timer1);
    convolve( Simg_gauss, ShData, gauss,  height, width, k_size, 'g');

    convolve( Simg_gx, Simg_gauss, sobel_x,  height , width, k_size, 'x');
    
    convolve( Simg_gy, Simg_gauss, sobel_y,  height , width, k_size, 'y');
    
    GetMag(Simg_gMag, Simg_gx, Simg_gy, height/comm_size, width);
	
    GetDir(Simg_gDir, Simg_gx, Simg_gy, height/comm_size, width);
	
    NMSuppress(Simg_nms, Simg_gMag, Simg_gx, Simg_gy, 0.17, 0.199, width, height/comm_size);

    Hysteresis(Simg_fOut, Simg_nms, Simg_gx, Simg_gy, 0.15, 0.199, width, height/comm_size);





    MPI_Gather(ShData, width*(height/comm_size), MPI_FLOAT, hData,
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	
    MPI_Gather(Simg_gauss, width*(height/comm_size), MPI_FLOAT, img_gauss,
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
   
    MPI_Gather(Simg_gx, width*(height/comm_size), MPI_FLOAT, img_gx,
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	  
 	MPI_Gather(Simg_gy, width*(height/comm_size), MPI_FLOAT, img_gy,
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	
    	MPI_Gather(Simg_gMag, width*(height/comm_size), MPI_FLOAT, img_gMag,
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	
    MPI_Gather(Simg_gDir, width*(height/comm_size), MPI_FLOAT, img_gDir,
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	
    MPI_Gather(Simg_nms, width*(height/comm_size), MPI_FLOAT, img_nms,
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    MPI_Gather(Simg_fOut, width*(height/comm_size), MPI_FLOAT, img_fOut,
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);

   //if(ID==0){

    
printf("IM HERE\n\n\n\n\n");
    MPI_Barrier(MPI_COMM_WORLD);
    

    
   if(ID ==0){

  sdkStopTimer(&timer1);
    printf("PROCES time: %f (ms)\n", sdkGetTimerValue(&timer1));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer1) / 1000.0f)) / 1e6);
    

    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_gauss.pgm");
    sdkSavePGM(outputFilename, img_gauss, width, height);
    printf("Wrote '%s'\n", outputFilename);

    //char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_mag.pgm");
    sdkSavePGM(outputFilename, img_gMag, width, height);
    printf("Wrote '%s'\n", outputFilename);

    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_nms.pgm");
    sdkSavePGM(outputFilename, img_nms, width, height);
    printf("Wrote '%s'\n", outputFilename);

    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_hys.pgm");
    sdkSavePGM(outputFilename, img_fOut, width, height);
    printf("Wrote '%s'\n", outputFilename);


    //printf("Hello World!");

	  sdkStopTimer(&timer);
    printf("Total time: %f (ms)\n", sdkGetTimerValue(&timer));
	printf("OVERHEAD : %f (ms)" ,sdkGetTimerValue(&timer)-sdkGetTimerValue(&timer1));
    sdkDeleteTimer(&timer);	
	sdkDeleteTimer(&timer1);
	
	
	
	}

     MPI_Finalize();
	return 0;


}

//functions

void createfilt2(char a, float* arr, int size){
if (a == 'x'){
  arr[0*size +0] = -1;
  arr[0*size +1] = 0;
  arr[0*size +2] = 1;
  arr[1*size +0] = -2;
  arr[1*size +1] = 0;
  arr[1*size +2] = 2;
  arr[2*size +0] = -1;
  arr[2*size +1] = 0;
  arr[2*size +2] = 1;  
}
else if(a == 'y'){
  arr[0*size +0] = -1;
  arr[0*size +1] = -2;
  arr[0*size +2] = -1;
  arr[1*size +0] = 0;
  arr[1*size +1] = 0;
  arr[1*size +2] = 0;
  arr[2*size +0] = 1;
  arr[2*size +1] = 2;
  arr[2*size +2] = 1;
}

else{
  arr[0*size +0] = 0.0625;
  arr[0*size +1] = 0.125;
  arr[0*size +2] = 0.0625;
  arr[1*size +0] = 0.125;
  arr[1*size +1] = 0.25;
  arr[1*size +2] = 0.125;
  arr[2*size +0] = 0.0625;
  arr[2*size +1] = 0.125;
  arr[2*size +2] = 0.0625;
}


}

void convolve(float* output, float* input, float* filt, int height, int width, int k_size, char type){


 for (int m = 0; m < height; m++ ) { //i
   for (int n = 0; n < width; n++ ) {//j
   float accumulation = 0;
   //float weightsum = 0;
   int val = k_size/2;
   for (int i = m-(val); i < (m+val+1); i++ ) {//k
     for (int j = n-(val); j < (n+val+1); j++ ) {//m
            if(i>=0 && i<width-1 && j>=0 && j< height-1){
       float k = input[(i)*height + (j)];//input(m+i, n+j); right by a b
       accumulation = accumulation + (k * filt[(i+val-m)*(k_size) + (j+val-n)]);//filt[(i+1)*(val) + (j+1)];//accumulation += k * kernel[1+i][1+j];
     
       }

     }
   }

    if (accumulation > 1.0) {
     accumulation = 1.0;
	}
    if (accumulation < 0.0) {
     accumulation = 0.0;
	}
    
    output[m*height + n] = accumulation;

   }
 }

}


void GetMag(float* output, float* imx, float* imy, int height, int width){
  for (int m = 0; m < height; m++ ) { //i
    for (int n = 0; n < width; n++ ) {//j
      float accumulation = sqrt(imx[m*height + n]*imx[m*height + n] + imy[m*height + n]*imy[m*height + n]); 
      output[m*height + n] = accumulation;

    }
  }
}

void GetDir(float* output, float* imx, float* imy, int height, int width){
  for (int m = 0; m < height; m++ ) {
    for (int n = 0; n < width; n++ ) {
      float theta = atan(imy[m*height + n]/imx[m*height + n]); 
      output[m*height + n] = theta;

    }
  }
}

void NMSuppress(float* output, float* G,float* after_Gx, float* after_Gy, float thr_min, float thr_max, int width, int height){

int nx = width;
int ny = height;
// Non-maximum suppression, straightforward implementation.
    for (int i = 1; i < nx - 1; i++){
        for (int j = 1; j < ny - 1; j++) {
            const int c = i + nx * j;
            const int nn = c - nx;
            const int ss = c + nx;
            const int ww = c + 1;
            const int ee = c - 1;
            const int nw = nn + 1;
            const int ne = nn - 1;
            const int sw = ss + 1;
            const int se = ss - 1;
 
            const float dir = (float)(fmod(atan2(after_Gy[c], after_Gx[c]) + M_PI, M_PI) / M_PI)*8;
 
            if (((dir <= 1 || dir > 7) && G[c] > G[ee] &&
                 G[c] > G[ww]) || // 0 deg
                ((dir > 1 && dir <= 3) && G[c] > G[nw] &&
                 G[c] > G[se]) || // 45 deg
                ((dir > 3 && dir <= 5) && G[c] > G[nn] &&
                 G[c] > G[ss]) || // 90 deg
                ((dir > 5 && dir <= 7) && G[c] > G[ne] &&
                 G[c] > G[sw]))   // 135 deg
                output[c] = G[c];
            else
                output[c] = 0;
        }
     }

} 
void Hysteresis(float* output, float* nms,float* after_Gx, float* after_Gy, float thr_min, float thr_max, int width, int height){
    int nx = width;
    int ny = height;
    // Reuse array
    // used as a stack. nx*ny/2 elements should be enough.
    int *edges = (int*) after_Gy;
    memset(output, 0, sizeof(float) * nx * ny);
    memset(edges, 0, sizeof(float) * nx * ny);
 
    // Tracing edges with hysteresis . Non-recursive implementation.
    size_t c = 0;
    for (int j = 1; j < ny - 1; j++){
        for (int i = 1; i < nx - 1; i++) {
            if (nms[c] >= thr_max && output[c] == 0.0) { // trace edges
    //printf("NOT HERE 3 \n");
                output[c] = 1.0;
                int nedges = 1;
                edges[0] = c;
 			    //printf("not here1");
                do {
                    nedges--;
                    const int t = edges[nedges];
 
                    int nbs[8]; // neighbours
                    nbs[0] = t - nx;     // n
                    nbs[1] = t + nx;     // s
                    nbs[2] = t + 1;      // w
                    nbs[3] = t - 1;      // e
                    nbs[4] = nbs[0] + 1; // nw
                    nbs[5] = nbs[0] - 1; // ne
                    nbs[6] = nbs[1] + 1; // sw
                    nbs[7] = nbs[1] - 1; // se
                
                    for (int k = 0; k < 8; k++)
                        if (nms[nbs[k]] >= thr_min && output[nbs[k]] == 0.0) {
			    //printf("am here! ");
                            output[nbs[k]] = 1.0;
                            edges[nedges] = nbs[k];
                            nedges++;
                        }
                } while (nedges > 0);
            }
            c++;
        }
    }
    //free(after_Gx);
    //free(after_Gy);
    //free(G);
    //free(nms);



}
