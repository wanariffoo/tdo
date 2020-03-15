#include<iostream>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n)
  {
    y[i] = a*x[i] + y[i];       // 16
    
    // for ( int i = 0 ; i < 1 ; i++)
    // {
    //     x[i] = x[i] + 1;       // 8*10
    //     // for ( int j = 0 ; j < 2000 ; j++ )
    //     //     x[i] += 2;
    // }
  } 
      

}

int main(void)
{
  int N = 30 * (1 << 20);
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

    std::cout << "N = " << N << std::endl;

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+511)/512, 512>>>(N, 2.0f, d_x, d_y);

  cudaEventRecord(stop);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i]-4.0f));
  }

  printf("Max error: %f\n", maxError);
  printf("Effective Bandwidth (GB/s): %f\n", N*4*(2)/milliseconds/1e6);
  std::cout << "Time taken = " << milliseconds << " ms" << std::endl;
}