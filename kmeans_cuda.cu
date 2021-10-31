#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

// Include extra helper functions for CUDA - handles image load/save
#include <cuda_runtime.h>
#include <helper_functions.h>    
#include <helper_cuda.h>         

const int k = 128;
const char *imageFilename = "suburb.ppm";

__device__ float cTemp[k*3];
__device__ float count[k];
__device__ bool stop=false;

__global__ void function(unsigned char *in_data,unsigned char *o_data, unsigned int width, unsigned int height,float * clus){
  int i   = threadIdx.y + blockDim.y*blockIdx.y;
  int j   = threadIdx.x + blockDim.x*blockIdx.x;

  float newsum, r, g, b = 0.0;

  float sum = 99999.0;
  int cVal = 0;
  int idx = i*width*4+j*4;

  unsigned int size = k;
  for(int a = 0; a < size; a++){
    newsum = abs(clus[a*3]-in_data[idx])+abs(clus[a*3+1]-in_data[idx+1])+abs(clus[a*3+2]-in_data[idx+2]);
    if(sum > newsum){
      cVal = a;
      r = (in_data[idx]);
      g = (in_data[idx+1]);
      b = (in_data[idx+2]);
      o_data[idx] = a;
      o_data[idx+1]=a;
      o_data[idx+2]=a;
      sum = newsum;
    }
  }

  /*count[cVal]++;
  cTemp[cVal*3] +=r;
  cTemp[cVal*3+1] +=g;
  cTemp[cVal*3+2] +=b;*/
  atomicAdd(&count[cVal],1.0);
  atomicAdd(&cTemp[cVal],r);
  atomicAdd(&cTemp[cVal+1],g);
  atomicAdd(&cTemp[cVal+2],b);
  __syncthreads();
  }


__global__ void function2(unsigned char *in_data,unsigned char *o_data, unsigned int width, unsigned int height,float * clus){
  int i=threadIdx.y + blockDim.y*blockIdx.y;
  int j=threadIdx.x + blockDim.x*blockIdx.x;
  int idx = i*width*4+j*4;
  o_data[idx]    = clus[o_data[idx]*3];
  o_data[idx+1]  = clus[o_data[idx+1]*3+1];
  o_data[idx+2]  = clus[o_data[idx+2]*3+2];
}


int main(int argc, char **argv){
  //timer for the entire program 
  StopWatchInterface *Ov_timer=NULL;
  sdkCreateTimer(&Ov_timer);
  sdkStartTimer(&Ov_timer);

  //loading input image
  char *imagePath = sdkFindFilePath(imageFilename, "");
  if (imagePath == NULL){
      printf("Unable to source image file: %s\n", imageFilename);
      exit(EXIT_FAILURE);
  }

  unsigned int width, height;
  unsigned char *hData = NULL;
  unsigned char *dData = NULL;
  //unsigned char *dOut = NULL;
  unsigned char *dOut = NULL;
  float *dClus = NULL;

  char outputFilename[1024];

  sdkLoadPPM4ub(imagePath, &hData, &width, &height);

  int csize = k;
  unsigned int size = width * height * sizeof(unsigned char)*4;
  unsigned char * hOutputData = (unsigned char *)malloc(size);

  //array 4 cluster centres on host and 4 update
  float* clusterH = (float *)malloc(csize*3* sizeof(float));
  float* clusterT = (float *)malloc(csize*3* sizeof(float));
  float* counter = (float *)malloc(csize* sizeof(float));

  cudaMalloc((void **) &dData, size);
  cudaMalloc((void **) &dOut, size);
  cudaMalloc((void **) &dClus, csize*3*sizeof(float));

  //initialize random cluster centers

  //int randomnumber, randomnumber1, randomnumber2;
  for(int i=0;i<csize;i++){
    /*int randomnumber = rand() % height*width;
    int randomnumber1 = rand() % height*width;
    int randomnumber2 = rand() % height*width;

    clusterH[i*3]=hData[randomnumber*4];
    clusterH[i*3+1]=hData[randomnumber1*4+1];
    clusterH[i*3+2]=hData[randomnumber2*4+2];*/
    int randomnum;
    randomnum = rand() % width*height;
    clusterH[i*3]    = hData[randomnum*4];//*1];
    clusterH[i*3+1]  = hData[randomnum*4+1];//*2+1];
    clusterH[i*3+2]  = hData[randomnum*4+2];//*3+2];
  }

cudaMemcpy(dData, hData, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dClus, clusterH , csize*3*sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  bool stop = false;
  while(stop == false){
    function<<<dimGrid,dimBlock>>>(dData, dOut, width, height, dClus);
    cudaDeviceSynchronize();
    float error=0;

    cudaMemcpyFromSymbol(clusterT, cTemp, csize*sizeof(float)*3, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(counter, count, csize*sizeof(float), 0, cudaMemcpyDeviceToHost);

      for(int a = 0; a < csize; a++){
        if(counter[a] != 0){
          error += abs(clusterH[a*3]-clusterT[a*3]/counter[a])+abs(clusterH[a*3+1]-clusterT[a*3+1]/counter[a])+abs(clusterH[a*3+2]-clusterT[a*3+2]/counter[a]);
        }
      }

      if(error/csize<10){
        stop=true;
      }

      for(int s=0;s<csize;s++){
        if(counter[s]!=0){
          clusterH[s*3]=clusterT[s*3]/counter[s];
          clusterH[s*3+1]=clusterT[s*3+1]/counter[s];
          clusterH[s*3+2]=clusterT[s*3+2]/counter[s];
        }
        clusterT[s*3]=0.0;
        clusterT[s*3+1]=0.0;
        clusterT[s*3+2]=0.0;
        counter[s]=0;
      }

      cudaMemcpyToSymbol(dClus, clusterH , csize*3*sizeof(float),0, cudaMemcpyHostToDevice);
      cudaMemcpyToSymbol(cTemp, clusterT , csize*3*sizeof(float),0, cudaMemcpyHostToDevice);
      cudaMemcpyToSymbol(count, counter , csize*sizeof(float),0, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();

  }

  function2<<<dimGrid,dimBlock>>>(dData, dOut, width, height, dClus);
  cudaDeviceSynchronize();

  sdkStopTimer(&timer);

  cudaMemcpy(hOutputData, dOut, size, cudaMemcpyDeviceToHost);

  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_out_cuda.ppm");
  sdkSavePPM4ub(outputFilename, hOutputData, width, height);
  sdkStopTimer(&Ov_timer);

  printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  printf("%.2f Mpixels/sec\n", (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
  printf("Overhead time: %f (ms)\n",(sdkGetTimerValue(&Ov_timer) - sdkGetTimerValue(&timer)));
   printf("Total time: %f (ms)\n", sdkGetTimerValue(&Ov_timer));
  sdkDeleteTimer(&timer);
  sdkDeleteTimer(&Ov_timer);

  return 0;
}
