#ifndef TRTAPI_H
#define TRTAPI_H

#include <NvInfer.h>
#include <iostream>
#include <vector>
#include <map>
using namespace std;
using namespace nvinfer1;

class trtAPI
{
public:
    trtAPI();

    void print_Weights(Weights wgt, int wgt_size);
    ITensor* trt_input(INetworkDefinition* network, const char* inputName,int C,int H,int W);
    ITensor* trt_conv(INetworkDefinition* network, ITensor* input,vector<float> wgt,vector<float> bias,int outputC,
                      int kernel = 1,int stride = 1,int padding = 0, int group =1,int dilations = 1);
    ITensor* trt_pool(INetworkDefinition* network, ITensor*input, PoolingType pType,int kernel = 1,int stride = 1,int padding = 0);
    ITensor* trt_batchNorm(INetworkDefinition* network, ITensor*input, vector<float> alpha, vector<float> gamma,vector<float> mean,vector<float> var,
                           float eps = 1.0e-5);
    ITensor* trt_activ(INetworkDefinition* network, ITensor* input,ActivationType Type,float alpha = 0.0,float beta = 0.0);
    //FullyConnected
    ITensor* trt_FC(INetworkDefinition* network, ITensor* input, int outputC, vector<float> wgt, vector<float> bias);
    ITensor* trt_elt(INetworkDefinition* network, ITensor* input1, ITensor* input2, ElementWiseOperation Type);
    ITensor* trt_concat(INetworkDefinition* network,vector<ITensor*> inputs, int axis = 0);
    ITensor* trt_slice(INetworkDefinition* network , ITensor* input, Dims start,Dims size,Dims step);
    ITensor* trt_softmax(INetworkDefinition* network, ITensor* input);
    ITensor* trt_MatMul(INetworkDefinition* network, ITensor* input1, bool transpose1,ITensor* input2,bool transpose2);

    ITensor* trt_shuffle_reshape(INetworkDefinition* network, ITensor* input,Dims dims);  //just reshape
    ITensor* trt_shuffle_premute(INetworkDefinition* network, ITensor* input, Permutation pmt);  //just permute
    ITensor* trt_shuffle(INetworkDefinition* network, ITensor* input, Dims dims,Permutation pmt, bool Reshape_first); //reshape & permute
    ITensor* trt_reduce(INetworkDefinition* network, ITensor* input, uint32_t reduceAxes,ReduceOperation op,bool keepDims = false);
    ITensor* trt_topK(INetworkDefinition* network, ITensor* input, int k = 1, TopKOperation op = TopKOperation::kMAX ,
                      uint32_t reduceAxes = 0x01,int outputAxes = 0);
    ITensor* trt_upsample(INetworkDefinition* network, ITensor* input, ResizeMode mode, Dims outdims);

    //The most important layer!!!!
    ITensor* trt_constant(INetworkDefinition* network, Dims dims, vector<float> wgt);

    ITensor* trt_PReLU(INetworkDefinition* network, ITensor* input,vector<float> wgt);
    ITensor* trt_groupNorm(INetworkDefinition* network,ITensor* input, int groups,vector<float> wgt,vector<float> bias,float eps = 1e-5);
	ITensor* trt_padding(INetworkDefinition* network,ITensor* input,DimsHW prePad = DimsHW{0,0},DimsHW postPad = DimsHW{0,0});


};

#endif // TRTAPI_H
