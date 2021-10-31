#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

// Include extra helper functions for CUDA - handles image load/save
#include <cuda_runtime.h>
#include <helper_functions.h>  
#include <helper_cuda.h>       

const int k = 100;

const char *imageFilename = "storm.ppm";

void serial(unsigned char *hdata,unsigned char *outdata, unsigned int width, unsigned int height,float * clus){

  int idx;
  int size = k;
  float cls_Temp [k*3]; //linear array for centres for each r g b
  float count[k];
  while(true){
    //initializes the temporary cluster array and the count array 
    for(int i = 0; i < size; i++){

      cls_Temp[i*3] = 0;
      cls_Temp[i*3+1] = 0;
      cls_Temp[i*3+2] = 0;
      count[i] = 0;
    }

    for(int i = 0; i < height; i++){
      for(int j = 0; j < width; j++){

        float newsum, r, g, b = 0.0;
        float sum = 99999.0;
        int cVal = 0;
        int idx = i*width*4+j*4;
        for(int a = 0; a < size; a++){
          newsum=abs(clus[a*3]-hdata[idx])+abs(clus[a*3+1]-hdata[idx+1])+abs(clus[a*3+2]-hdata[idx+2]);
          if(sum > newsum){
            cVal = a;
            r = (hdata[idx]);
            g = (hdata[idx+1]);
            b = (hdata[idx+2]);
            outdata[idx]=a;
            outdata[idx+1]=a;
            outdata[idx+2]=a;
            sum = newsum;
          }
        }

        count[cVal]++;
        cls_Temp[cVal*3] +=r;
        cls_Temp[cVal*3+1] +=g;
        cls_Temp[cVal*3+2] +=b;
   }
  }

  float error = 0.0;
  for(int a = 0; a < size; a++){
    if(count[a] != 0.0){
      error += abs(clus[a*3]-(cls_Temp[a*3]/count[a]))+abs(clus[a*3+1]-(cls_Temp[a*3+1]/count[a]))+abs(clus[a*3+2]-(cls_Temp[a*3+2]/count[a]));
    }
  }

  if((error/(float)size)<10){
    //stop=true;
    printf("%d\n",size );
    break;
  }

  for(int s = 0; s < k; s++){
    if(count[s] != 0.0){
    clus[s*3]   = cls_Temp[s*3]/count[s];
    clus[s*3+1] = cls_Temp[s*3+1]/count[s];
    clus[s*3+2] = cls_Temp[s*3+2]/count[s];
  }
  cls_Temp[s*3]    = 0;
  cls_Temp[s*3+1]  = 0;
  cls_Temp[s*3+2]  = 0;
  count[s] = 0;
  }

  }


  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      idx = i*width*4+j*4;
      outdata[idx]    = clus[outdata[idx]*3];
      outdata[idx+1]  = clus[outdata[idx+1]*3+1];
      outdata[idx+2]  = clus[outdata[idx+2]*3+2];
    }
  }


  
}


int main(int argc, char **argv){
  //timer for entire program 
  StopWatchInterface *timer =NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  //loading the image
  char *imagePath = sdkFindFilePath(imageFilename, "");
  if (imagePath == NULL){
      printf("Unable to source image file: %s\n", imageFilename);
      exit(EXIT_FAILURE);
  }

  unsigned int width, height;
  unsigned char *hData = NULL;
  //unsigned char *dOut = NULL;
  char outputFilename[1024];

  sdkLoadPPM4ub(imagePath, &hData, &width, &height);

  int csize = k;
  unsigned int size = width * height * sizeof(unsigned char)*4;
  unsigned char * hOutputData = (unsigned char *)malloc(size);
  //array for all cluster centers
  float* cluster = (float *)malloc(csize*3* sizeof(float));
  //float* counter  = (float *)malloc(csize* sizeof(float));
  //initialize random cluster centers
  int randomnumber, randomnumber1, randomnumber2;
  for(int i=0;i<csize;i++){
    randomnumber = rand() % height*width;
    randomnumber1 = rand() % height*width;
    randomnumber2 = rand() % height*width;

    cluster[i*3]=hData[randomnumber*4];
    cluster[i*3+1]=hData[randomnumber1*4+1];
    cluster[i*3+2]=hData[randomnumber2*4+2];
    /*cluster[i*3]=hData[randomnumber*4];
    cluster[i*3+1]=hData[randomnumber1*4+1];
    cluster[i*3+2]=hData[randomnumber2*4+2];*/
  }

  StopWatchInterface *timer1 = NULL;
  sdkCreateTimer(&timer1);
  sdkStartTimer(&timer1);
  serial(hData, hOutputData, width, height, cluster);
  sdkStopTimer(&timer1);

 // cudaMemcpy(hOutputData, dOut, size, cudaMemcpyDeviceToHost);

  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_out_sr.ppm");
  sdkSavePPM4ub(outputFilename, hOutputData, width, height);
  sdkStopTimer(&timer);

  //float ov_time = (sdkGetTimerValue(&timer1)- sdkGetTimerValue(&timer));//NULL;
  printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer1));
  printf("%.2f Mpixels/sec\n", (width *height / (sdkGetTimerValue(&timer1) / 1000.0f)) / 1e6);
  printf("Overhead time: %f (ms)\n",(sdkGetTimerValue(&timer) - sdkGetTimerValue(&timer1)));
   printf("Total time: %f (ms)\n", sdkGetTimerValue(&timer));
  //printf("Overhead time: %f (ms)\n", (sdkGetTimerValue(&timer)- sdkGetTimerValue(&timer1)));

  sdkDeleteTimer(&timer);
  sdkDeleteTimer(&timer1);
  return 0;
}
