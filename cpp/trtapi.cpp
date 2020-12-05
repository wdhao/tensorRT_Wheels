#include "trtapi.h"
#include <assert.h>
#include <fstream>


trtAPI::trtAPI()
{

}

/*print wgt values
 * e.g!
 * print_Weights(wgt,wgt_size);
*/
void trtAPI::print_Weights(Weights wgt, int wgt_size)
{
    for (int i = 0;i < wgt_size; i++) {
        cout<<*((float*)wgt.values + i)<<endl;
    }
}
//e.g! auto input = trt_input(m_network,"data",3,512,512);
ITensor* trtAPI::trt_input(INetworkDefinition *network, const char *inputName, int C, int H, int W)
{
    ITensor* input = network->addInput(inputName,DataType::kFLOAT,Dims3{C,H,W});
    assert(input);
    return input;
}
/*default : group = 1, dilations = 1
 * e.g!
 * auto convlayer = trt_conv(m_networ, *input, wgt, bias, 64, 3, 1, 1);
*/
ITensor* trtAPI::trt_conv(INetworkDefinition *network, ITensor* input, vector<float> wgt,vector<float> bias, int outputC,
                          int kernel, int stride, int padding, int group, int dilations)
{

    int Size = input->getDimensions().d[0] * outputC * kernel * kernel;
    int size = wgt.size();
    if(Size != size)
    {
        cout<< "load weights error!"<<endl;
        assert(0);
    }
    nvinfer1::Weights convWeights {nvinfer1::DataType::kFLOAT,nullptr,Size};
    float *weights = new float[Size];
    for (int i = 0; i < size; i++) {
        weights[i] = wgt.at(i);
    }
    convWeights.values = weights;
    nvinfer1::Weights convBias {nvinfer1::DataType::kFLOAT,nullptr,outputC};
    float *var = new float[outputC];
    for (int i = 0; i < outputC; i++) {
        var[i] = 0.0;
        if(bias.size() != 0)
            var[i] = bias.at(i);
    }
    convBias.values = var;
    auto conv =  network->addConvolutionNd(*input,outputC,DimsHW{kernel,kernel},convWeights,convBias);
    assert(conv);
    conv->setStrideNd(DimsHW{stride,stride});
    conv->setPaddingNd(DimsHW{padding,padding});
    conv->setDilationNd(DimsHW{dilations,dilations});
    conv->setNbGroups(group);
    return conv->getOutput(0);
}
/*e.g!
 * auto poollayer = trt_pool(m_network, *input, PoolingType::kMAX, 2, 2, 0);
*/
ITensor* trtAPI::trt_pool(INetworkDefinition *network, ITensor *input, PoolingType pType, int kernel, int stride, int padding)
{
    auto pool = network->addPoolingNd(*input,pType,DimsHW{kernel,kernel});
    assert(pool);
    pool->setStrideNd(DimsHW{stride,stride});
    pool->setPaddingNd(DimsHW{padding,padding});
    return pool->getOutput(0);
}
/*e.g!
 * auto batchNorm = trt_batchNorm(network,input,weights,bias,mean,var)
*/
ITensor* trtAPI::trt_batchNorm(INetworkDefinition *network, ITensor *input, vector<float> weights, vector<float> bias,vector<float> mean,vector<float> var,float eps)
{
    unsigned int size = bias.size();
    if(weights.size()==0 || size ==0 || mean.size() ==0 || var.size() ==0)
    {
        cout<<"load weights error! Please check it!"<<endl;
        assert(0);
    }
    Weights scale{DataType::kFLOAT,nullptr,size};
    Weights shift{DataType::kFLOAT,nullptr,size};
    Weights power{DataType::kFLOAT,nullptr,size};
    vector<float>bn_var;
    for (int i = 0; i < size; i++) {
        bn_var.push_back( sqrt(var.at(i) + eps));
    }
    float *shiftWt = new float[size];
    for (int i = 0; i < size; i++) {
        shiftWt[i] = bias.at(i)-(mean.at(i)*weights.at(i)/bn_var.at(i));
    }
    shift.values = shiftWt;
    float *scaleWt = new float[size];
    float *powerWt = new float[size];
    for (int i = 0; i <size; i++) {
        scaleWt[i] = weights.at(i)/bn_var.at(i);
        powerWt[i] = 1.0;
    }
    scale.values = scaleWt;
    power.values = powerWt;
    ScaleMode scaleMode = ScaleMode::kCHANNEL;
    // output = ( input * scale +  shift)^ power
    IScaleLayer *batchNorm = network->addScale(*input,scaleMode,shift,scale,power);
    assert(batchNorm);
    return batchNorm->getOutput(0);
}
/*e.g!
 * auto activ = trt_activ(network, input, ActivationType::kRELU);
*/
ITensor* trtAPI::trt_activ(INetworkDefinition *network, ITensor *input, ActivationType Type, float alpha, float beta)
{
    IActivationLayer* activ = network->addActivation(*input,Type);
    switch(Type)
    {
    case ActivationType::kLEAKY_RELU :          //x>=0 ? x : alpha * x.
        activ->setAlpha(alpha);
        break;
    case ActivationType::kELU :                 //x>=0 ? x : alpha * (exp(x) - 1).
        activ->setAlpha(alpha);
        break;
    case ActivationType::kSELU :                //x>0 ? beta * x : beta * (alpha*exp(x) - alpha)
        activ->setAlpha(alpha);
        activ->setBeta(beta);
        break;
    case ActivationType::kSOFTPLUS :            //alpha*log(exp(beta*x)+1)
        activ->setAlpha(alpha);
        activ->setBeta(beta);
        break;
    case ActivationType::kCLIP :                //max(alpha, min(beta, x))   tips: alpha < beta
        activ->setAlpha(alpha);
        activ->setBeta(beta);
        break;
    case ActivationType::kHARD_SIGMOID :        //max(0, min(1, alpha*x+beta))
        activ->setAlpha(alpha);
        activ->setBeta(beta);
        break;
    case ActivationType::kSCALED_TANH :         //alpha*tanh(beta*x)
        activ->setAlpha(alpha);
        activ->setBeta(beta);
        break;
    case ActivationType::kTHRESHOLDED_RELU :    //x>alpha ? x : 0
        activ->setAlpha(alpha);
        break;
    default:
        break;

    }
    return activ->getOutput(0);
}
ITensor* trtAPI::trt_FC(INetworkDefinition *network, ITensor *input, int outputC, vector<float> wgt, vector<float> bias)
{
    int size = wgt.size();
    int Size = input->getDimensions().d[0] * outputC;
    if(Size != size)
    {
        cout<<"load weights error!"<<endl;
        assert(0);
    }
    Weights fcWgt{DataType::kFLOAT,nullptr,Size};
    Weights fcBias{DataType::kFLOAT,nullptr,outputC};
    float *FC_wgt = new float[Size];
    for (int i =0;i<Size;i++)
    {
        FC_wgt[i] = wgt.at(i);
    }
    fcWgt.values = FC_wgt;
    float *FC_bias = new float[outputC];
    if(bias.size() == outputC)
    {
        for (int j = 0;j<outputC;++j)
        {
            FC_bias[j] = bias.at(j);
        }
    }
    else{
        for (int j = 0;j<outputC;++j) {
            FC_bias[j] = 0.0;
        }
    }
    fcBias.values = FC_bias;
    IFullyConnectedLayer *fc = network->addFullyConnected(*input,outputC,fcWgt,fcBias);
    return fc->getOutput(0);
}
/*e.g!
 * output = input1 + input2
 * auto out = trt_elt(network,input1,input2,ElementWiseOperation::kSUM);
*/
ITensor* trtAPI::trt_elt(INetworkDefinition *network, ITensor* input1, ITensor* input2, ElementWiseOperation Type)
{
    IElementWiseLayer *elt = network->addElementWise(*input1,*input2,Type);
    return elt->getOutput(0);
}
/**/
ITensor* trtAPI::trt_concat(INetworkDefinition *network, vector<ITensor *> inputs, int axis)
{
    int num_inputs = inputs.size();
    if(num_inputs == 0)
    {
        cout<<"no inputs !"<<endl;
        assert(0);
    }
    ITensor** input = new ITensor*[num_inputs];
    for(int i = 0;i<num_inputs; i++)
    {
        input[i] = inputs[i];
    }
    IConcatenationLayer *cat = network->addConcatenation(input,num_inputs);
    cat->setAxis(axis);
    return cat->getOutput(0);

}
ITensor* trtAPI::trt_slice(INetworkDefinition *network, ITensor *input, Dims start, Dims size, Dims step)
{
    ISliceLayer *slice = network->addSlice(*input,start,size,step);
    return slice->getOutput(0);
}
ITensor* trtAPI::trt_softmax(INetworkDefinition *network, ITensor *input)
{
    ISoftMaxLayer *softmax = network->addSoftMax(*input);
    return softmax->getOutput(0);
}
ITensor* trtAPI::trt_MatMul(INetworkDefinition *network, ITensor *input1, bool transpose1, ITensor *input2, bool transpose2)
{
    IMatrixMultiplyLayer *mm = network->addMatrixMultiply(*input1,transpose1,*input2,transpose2);
    return mm->getOutput(0);
}
ITensor* trtAPI::trt_shuffle_reshape(INetworkDefinition *network, ITensor *input, Dims dims)
{
    IShuffleLayer *reshape = network->addShuffle(*input);
    reshape->setReshapeDimensions(dims);
    return reshape->getOutput(0);
}
ITensor* trtAPI::trt_shuffle_premute(INetworkDefinition *network, ITensor *input, Permutation pmt)
{
    IShuffleLayer *premute = network->addShuffle(*input);
    premute->setFirstTranspose(pmt);
    return premute->getOutput(0);
}
ITensor* trtAPI::trt_shuffle(INetworkDefinition *network, ITensor *input, Dims dims, Permutation pmt, bool Reshape_first)
{
    IShuffleLayer *shuffle = network->addShuffle(*input);
    if(Reshape_first)
    {
        shuffle->setReshapeDimensions(dims);
        shuffle->setSecondTranspose(pmt);
    }
    else {
        shuffle->setFirstTranspose(pmt);
        shuffle->setReshapeDimensions(dims);
    }
    return shuffle->getOutput(0);
}
ITensor* trtAPI::trt_reduce(INetworkDefinition *network, ITensor *input, uint32_t reduceAxes, ReduceOperation op,bool keepDims)
{
    /*
     *reduceAxes W    H    C
     * 0X01      0    0    1
     * 0X02      0    1    0
     * 0X03      0    1    1
     * 0X04      1    0    0
     * 0X05      1    0    1
     * 0X06      1    1    0
     * 0X07      1    1    1
    */
    IReduceLayer *reduce = network->addReduce(*input,op,reduceAxes,keepDims);
    return reduce->getOutput(0);
}
ITensor* trtAPI::trt_topK(INetworkDefinition *network, ITensor *input, int k, TopKOperation op, uint32_t reduceAxes, int outputAxes)
{
    ITopKLayer *topK = network->addTopK(*input,op,k,reduceAxes);
    //tips: getOutput(0)  == values  ; getOutput(1) == index;
    return topK->getOutput(outputAxes);
}
ITensor* trtAPI::trt_upsample(INetworkDefinition *network, ITensor *input, ResizeMode mode, Dims outdims)
{
    IResizeLayer *upSample = network->addResize(*input);
    upSample->setResizeMode(mode);
    upSample->setOutputDimensions(outdims);

    if(mode == ResizeMode::kNEAREST)
    {
        upSample->setAlignCorners(false); // tips!
    }
    else {
        upSample->setAlignCorners(true); // tips!
    }

    return upSample->getOutput(0);
}
ITensor* trtAPI::trt_constant(INetworkDefinition *network, Dims dims, vector<float> wgt)
{
    int len = wgt.size();
    Weights weights{DataType::kFLOAT,nullptr,len};
    float *var = new float[len];
    for(int i = 0; i < len; i++)
    {
        var[i] = wgt.at(i);
    }
    weights.values = var;
    IConstantLayer *cons = network->addConstant(dims,weights);
    return cons->getOutput(0);
}
ITensor* trtAPI::trt_PReLU(INetworkDefinition *network, ITensor* input,vector<float> wgt)
{
    int dimC = input->getDimensions().d[0];
    auto slope = trt_constant(network,Dims3{dimC,1,1},wgt);
    IParametricReLULayer *pReLU = network->addParametricReLU(*input,*slope);
    return pReLU->getOutput(0);
}
ITensor* trtAPI::trt_groupNorm(INetworkDefinition *network, ITensor *input, int groups, vector<float> wgt, vector<float> bias, float eps)
{
    int dimC = input->getDimensions().d[0];
    if(dimC % groups != 0)
    {
        cout<< "input channel % groups  != 0 "<<endl;
        assert(0);
    }
    Dims dims;
    dims.nbDims = 2;
    dims.d[0] = groups;
    dims.d[1] = -1;
    IShuffleLayer *shuffle1 = network->addShuffle(*input);
    shuffle1->setReshapeDimensions(dims);  //[c,h,w] -->[groups,c*h*w/groups]
    IReduceLayer *mean = network->addReduce(*shuffle1->getOutput(0),ReduceOperation::kAVG,2,true);//mean : [groups,1]
    IElementWiseLayer *elt_sub = network->addElementWise(*shuffle1->getOutput(0),*mean->getOutput(0),
                                                                     ElementWiseOperation::kSUB);   // x - mean
    IElementWiseLayer *elt_prod = network->addElementWise(*elt_sub->getOutput(0),*elt_sub->getOutput(0),
                                                                      ElementWiseOperation::kPROD); // var**2 = (x-mean)**2
    IReduceLayer *var_var =  network->addReduce(*elt_prod->getOutput(0),ReduceOperation::kSUM,2,true);//sum(var**2) :[groups,1]
    Weights E{DataType::kFLOAT,nullptr,groups};
    float* e = new float[groups];
    for (int i =0; i< groups; i++)
    {
        e[i] = 1e-5;
    }
    E.values = e;
    IConstantLayer *e_layer = network->addConstant(Dims2{groups,1},E);//1e-5
    float* length = new float[groups];
    for (int i =0; i< groups; i++)
    {
        length[i] = shuffle1->getOutput(0)->getDimensions().d[1]*1.0;
    }
    E.values = length;
    IConstantLayer *e_length = network->addConstant(Dims2{groups,1},E);//length = c*h*w/32
    IElementWiseLayer *elt_div = network->addElementWise(*var_var->getOutput(0),*e_length->getOutput(0),
                                                                     ElementWiseOperation::kDIV); //sum(var**2) / length
    IElementWiseLayer *elt_sum = network->addElementWise(*elt_div->getOutput(0),*e_layer->getOutput(0),
                                                                     ElementWiseOperation::kSUM); // 1e-5 + sum(var**2)/length
    float* sqrt = new float[groups];
    for (int i =0; i< groups; i++)
    {
        sqrt[i] = 0.5;
    }
    E.values = sqrt;
    IConstantLayer *e_sqrt = network->addConstant(Dims2{groups,1},E);//
    IElementWiseLayer *elt_sqrt = network->addElementWise(*elt_sum->getOutput(0),*e_sqrt->getOutput(0),
                                                                      ElementWiseOperation::kPOW);//var = sqrt(1e-5 + sum(var**2)/length)
    IElementWiseLayer *div_var = network->addElementWise(*elt_sub->getOutput(0),*elt_sqrt->getOutput(0),
                                                                     ElementWiseOperation::kDIV);//(x-mean)/var
    nvinfer1::IShuffleLayer *shuffle2 = network->addShuffle(*div_var->getOutput(0));//[groups,c*h*w/groups] -->[c,h,w]
    Dims dims2;
    dims2 = input->getDimensions();
    shuffle2->setReshapeDimensions(dims2);//[groups,c*h*w/groups] -->[c,h,w]
    Weights Scale{DataType::kFLOAT,nullptr,dimC};
    Weights Bias{nvinfer1::DataType::kFLOAT,nullptr,dimC};
    float *Scale_weight = new float[dimC];
    float *Bias_weight = new float[dimC];

    for(int i = 0;i<dimC;i++)
    {
        Scale_weight[i] = wgt[i];
        Bias_weight[i] = bias[i];
    }
    Scale.values = Scale_weight;
    Bias.values = Bias_weight;

    IConstantLayer *add_scale = network->addConstant(Dims3{dimC,1,1},Scale);//scale
    IElementWiseLayer *prod_scale = network->addElementWise(*shuffle2->getOutput(0),*add_scale->getOutput(0),
                                                                        ElementWiseOperation::kPROD);//x * scale
    IConstantLayer *add_bias = network->addConstant(Dims3{dimC,1,1},Bias);//bias
    IElementWiseLayer *sum_bias = network->addElementWise(*prod_scale->getOutput(0),*add_bias->getOutput(0),
                                                                      ElementWiseOperation::kSUM);//x * scale + bias
    return sum_bias->getOutput(0);

}
ITensor* trtAPI::trt_padding(INetworkDefinition *network, ITensor *input, DimsHW prePad, DimsHW postPad)
{
    IPaddingLayer *pad = network->addPaddingNd(*input,prePad,postPad);
    return pad->getOutput(0);
}

