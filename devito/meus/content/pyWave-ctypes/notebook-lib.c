#include <stddef.h>
    void cfun(const double *indatav, size_t size, double *outdatav){
        size_t i;
        for (i = 0; i < size; ++i)
            outdatav[i] = indatav[i] * 10.0;
    }
    void cfun32(const float *indatav, size_t size, float *outdatav){
        size_t i;
        for (i = 0; i < size; ++i)
            outdatav[i] = indatav[i] * 10.0;
    }
    float compute_mean(float* A, int n){
        int i;
        float sum = 0.0f;
        for(i=0; i<n; ++i)
            sum += A[i];
        return sum/n;
    }
    