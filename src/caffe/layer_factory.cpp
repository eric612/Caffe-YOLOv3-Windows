// Make sure we include Python.h before any system header
// to avoid _POSIX_C_SOURCE redefinition
#ifdef WITH_PYTHON_LAYER
#include <boost/python.hpp>
#endif
#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/detection_evaluate_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/layers/smooth_L1_loss_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/layers/yolov3_layer.hpp"
#include "caffe/layers/yolo_detection_output_layer.hpp"
#include "caffe/layers/yolov3_detection_output_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/reorg_layer.hpp"
#include "caffe/layers/depthwise_conv_layer.hpp"
#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/layers/cudnn_lcn_layer.hpp"
#include "caffe/layers/cudnn_lrn_layer.hpp"
#include "caffe/layers/cudnn_pooling_layer.hpp"
#include "caffe/layers/cudnn_relu_layer.hpp"
#include "caffe/layers/cudnn_sigmoid_layer.hpp"
#include "caffe/layers/cudnn_softmax_layer.hpp"
#include "caffe/layers/cudnn_tanh_layer.hpp"
#endif

#ifdef WITH_PYTHON_LAYER
#include "caffe/layers/python_layer.hpp"
#endif

namespace caffe {

template <typename Dtype>
typename LayerRegistry<Dtype>::CreatorRegistry&
LayerRegistry<Dtype>::Registry() {
  static CreatorRegistry* g_registry_ = new CreatorRegistry();
  return *g_registry_;
}

// Adds a creator.
template <typename Dtype>
void LayerRegistry<Dtype>::AddCreator(const string& type, Creator creator) {
  CreatorRegistry& registry = Registry();
  CHECK_EQ(registry.count(type), 0) << "Layer type " << type
                                    << " already registered.";
  registry[type] = creator;
}

// Get a layer using a LayerParameter.
template <typename Dtype>
shared_ptr<Layer<Dtype> > LayerRegistry<Dtype>::CreateLayer(
    const LayerParameter& param) {
  if (Caffe::root_solver()) {
    LOG(INFO) << "Creating layer " << param.name();
  }
  const string& type = param.type();
  CreatorRegistry& registry = Registry();
  CHECK_EQ(registry.count(type), 1)
      << "Unknown layer type: " << type
      << " (known types: " << LayerTypeListString() << ")";
  return registry[type](param);
}

template <typename Dtype>
vector<string> LayerRegistry<Dtype>::LayerTypeList() {
  CreatorRegistry& registry = Registry();
  vector<string> layer_types;
  for (typename CreatorRegistry::iterator iter = registry.begin();
       iter != registry.end(); ++iter) {
    layer_types.push_back(iter->first);
  }
  return layer_types;
}

// Layer registry should never be instantiated - everything is done with its
// static variables.
template <typename Dtype>
LayerRegistry<Dtype>::LayerRegistry() {}

template <typename Dtype>
string LayerRegistry<Dtype>::LayerTypeListString() {
  vector<string> layer_types = LayerTypeList();
  string layer_types_str;
  for (vector<string>::iterator iter = layer_types.begin();
       iter != layer_types.end(); ++iter) {
    if (iter != layer_types.begin()) {
      layer_types_str += ", ";
    }
    layer_types_str += *iter;
  }
  return layer_types_str;
}

template <typename Dtype>
LayerRegisterer<Dtype>::LayerRegisterer(
    const string& type,
    shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) {
  // LOG(INFO) << "Registering layer type: " << type;
  LayerRegistry<Dtype>::AddCreator(type, creator);
}

INSTANTIATE_CLASS(LayerRegistry);
INSTANTIATE_CLASS(LayerRegisterer);
/////////////////////////////////////////////////////

// Get Reorg layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetReorgLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new ReorgLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(Reorg, GetReorgLayer);

template <typename Dtype>
shared_ptr<Layer<Dtype> > GetSigmoidCrossEntropyLossLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new SigmoidCrossEntropyLossLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(SigmoidCrossEntropyLoss, GetSigmoidCrossEntropyLossLayer);

///////////////////////////////////////////////////////
// Get Flatten layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetAnnotatedDataLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new AnnotatedDataLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(AnnotatedData, GetAnnotatedDataLayer);


// Get BatchNorm layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetBatchNormLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new BatchNormLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(BatchNorm, GetBatchNormLayer);


// Get Bias layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetBiasLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new BiasLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(Bias, GetBiasLayer);


// Get Flatten layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetConcatLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new ConcatLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(Concat, GetConcatLayer);


// Get DetectionEvaluate layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetDetectionEvaluateLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new DetectionEvaluateLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(DetectionEvaluate, GetDetectionEvaluateLayer);


// Get Flatten layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetFlattenLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new FlattenLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(Flatten, GetFlattenLayer);


// Get input layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetInputLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new InputLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(Input, GetInputLayer);



// Get DetectionOutput layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetYoloDetectionOutputLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new YoloDetectionOutputLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(YoloDetectionOutput, GetYoloDetectionOutputLayer);

// Get DetectionOutput layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetYolov3DetectionOutputLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new Yolov3DetectionOutputLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(Yolov3DetectionOutput, GetYolov3DetectionOutputLayer);


// Get RegionLoss layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetRegionLossLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new RegionLossLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(RegionLoss, GetRegionLossLayer);
// Get Permute layer according to engine.

// Get RegionLoss layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetYolov3Layer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new Yolov3Layer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(Yolov3, GetYolov3Layer);


// Get Reshape layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetReshapeLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new ReshapeLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(Reshape, GetReshapeLayer);


// Get Scale layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetScaleLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new ScaleLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(Scale, GetScaleLayer);


// Get SoftmaxWithLoss layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetSoftmaxWithLossLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new SoftmaxWithLossLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(SoftmaxWithLoss, GetSoftmaxWithLossLayer);


// Get SmoothL1Loss layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetSmoothL1LossLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new SmoothL1LossLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(SmoothL1Loss, GetSmoothL1LossLayer);




// Get normalize layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetNormalizeLayer(const LayerParameter& param) {
	return shared_ptr<Layer<Dtype> >(new NormalizeLayer<Dtype>(param));
}
REGISTER_LAYER_CREATOR(Normalize, GetNormalizeLayer);

//////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

// Get convolution layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetConvolutionLayer(
    const LayerParameter& param) {
  ConvolutionParameter conv_param = param.convolution_param();
  ConvolutionParameter_Engine engine = conv_param.engine();
#ifdef USE_CUDNN
  bool use_dilation = false;
  for (int i = 0; i < conv_param.dilation_size(); ++i) {
    if (conv_param.dilation(i) > 1) {
      use_dilation = true;
    }
  }
#endif
  if (engine == ConvolutionParameter_Engine_DEFAULT) {
    engine = ConvolutionParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    if (!use_dilation) {
      engine = ConvolutionParameter_Engine_CUDNN;
    }
#endif
  }
  if (engine == ConvolutionParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new ConvolutionLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == ConvolutionParameter_Engine_CUDNN) {
    if (use_dilation) {
      LOG(FATAL) << "CuDNN doesn't support the dilated convolution at Layer "
                 << param.name();
    }
    return shared_ptr<Layer<Dtype> >(new CuDNNConvolutionLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);


// Get convolution layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetDepthwiseConvolutionLayer(
	const LayerParameter& param) {
	ConvolutionParameter conv_param = param.convolution_param();
	ConvolutionParameter_Engine engine = conv_param.engine();
#ifdef USE_CUDNN
	bool use_dilation = false;
	for (int i = 0; i < conv_param.dilation_size(); ++i) {
		if (conv_param.dilation(i) > 1) {
			use_dilation = true;
		}
	}
#endif
	if (engine == ConvolutionParameter_Engine_DEFAULT) {
		engine = ConvolutionParameter_Engine_CAFFE;
#ifdef USE_CUDNN
		if (!use_dilation) {
			engine = ConvolutionParameter_Engine_CUDNN;
		}
#endif
	}
	if (engine == ConvolutionParameter_Engine_CAFFE) {
		return shared_ptr<Layer<Dtype> >(new DepthwiseConvolutionLayer<Dtype>(param));
#ifdef USE_CUDNN
	}
	else if (engine == ConvolutionParameter_Engine_CUDNN) {
		if (use_dilation) {
			LOG(FATAL) << "CuDNN doesn't support the dilated convolution at Layer "
				<< param.name();
		}
		return shared_ptr<Layer<Dtype> >(new CuDNNConvolutionLayer<Dtype>(param));
#endif
	}
	else {
		LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
		throw;  // Avoids missing return warning
	}
}

REGISTER_LAYER_CREATOR(DepthwiseConvolution, GetDepthwiseConvolutionLayer);
// Get pooling layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetPoolingLayer(const LayerParameter& param) {
  PoolingParameter_Engine engine = param.pooling_param().engine();
  if (engine == PoolingParameter_Engine_DEFAULT) {
    engine = PoolingParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = PoolingParameter_Engine_CUDNN;
#endif
  }
  if (engine == PoolingParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == PoolingParameter_Engine_CUDNN) {
    if (param.top_size() > 1) {
      LOG(INFO) << "cuDNN does not support multiple tops. "
                << "Using Caffe's own pooling layer.";
      return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
    }
    // CuDNN assumes layers are not being modified in place, thus
    // breaking our index tracking for updates in some cases in Caffe.
    // Until there is a workaround in Caffe (index management) or
    // cuDNN, use Caffe layer to max pooling, or don't use in place
    // layers after max pooling layers
    if (param.pooling_param().pool() == PoolingParameter_PoolMethod_MAX) {
        return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
    } else {
        return shared_ptr<Layer<Dtype> >(new CuDNNPoolingLayer<Dtype>(param));
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer);

// Get LRN layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetLRNLayer(const LayerParameter& param) {
  LRNParameter_Engine engine = param.lrn_param().engine();

  if (engine == LRNParameter_Engine_DEFAULT) {
#ifdef USE_CUDNN
    engine = LRNParameter_Engine_CUDNN;
#else
    engine = LRNParameter_Engine_CAFFE;
#endif
  }

  if (engine == LRNParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new LRNLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == LRNParameter_Engine_CUDNN) {
    LRNParameter lrn_param = param.lrn_param();

    if (lrn_param.norm_region() ==LRNParameter_NormRegion_WITHIN_CHANNEL) {
      return shared_ptr<Layer<Dtype> >(new CuDNNLCNLayer<Dtype>(param));
    } else {
      // local size is too big to be handled through cuDNN
      if (param.lrn_param().local_size() > CUDNN_LRN_MAX_N) {
        return shared_ptr<Layer<Dtype> >(new LRNLayer<Dtype>(param));
      } else {
        return shared_ptr<Layer<Dtype> >(new CuDNNLRNLayer<Dtype>(param));
      }
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(LRN, GetLRNLayer);

// Get relu layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetReLU6Layer(const LayerParameter& param) {
	ReLUParameter_Engine engine = param.relu_param().engine();
	if (engine == ReLUParameter_Engine_DEFAULT) {
		engine = ReLUParameter_Engine_CAFFE;
#ifdef USE_CUDNN
		engine = ReLUParameter_Engine_CUDNN;
#endif
	}
	if (engine == ReLUParameter_Engine_CAFFE) {
		return shared_ptr<Layer<Dtype> >(new ReLULayer<Dtype>(param));
#ifdef USE_CUDNN
	}
	else if (engine == ReLUParameter_Engine_CUDNN) {
		return shared_ptr<Layer<Dtype> >(new CuDNNReLULayer<Dtype>(param));
#endif
	}
	else {
		LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
		throw;  // Avoids missing return warning
	}
}
REGISTER_LAYER_CREATOR(ReLU6, GetReLU6Layer);
// Get relu layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetReLULayer(const LayerParameter& param) {
  ReLUParameter_Engine engine = param.relu_param().engine();
  if (engine == ReLUParameter_Engine_DEFAULT) {
    engine = ReLUParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = ReLUParameter_Engine_CUDNN;
#endif
  }
  if (engine == ReLUParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new ReLULayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == ReLUParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNReLULayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(ReLU, GetReLULayer);

// Get sigmoid layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetSigmoidLayer(const LayerParameter& param) {
  SigmoidParameter_Engine engine = param.sigmoid_param().engine();
  if (engine == SigmoidParameter_Engine_DEFAULT) {
    engine = SigmoidParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = SigmoidParameter_Engine_CUDNN;
#endif
  }
  if (engine == SigmoidParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new SigmoidLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == SigmoidParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNSigmoidLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Sigmoid, GetSigmoidLayer);

// Get softmax layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetSoftmaxLayer(const LayerParameter& param) {
  SoftmaxParameter_Engine engine = param.softmax_param().engine();
  if (engine == SoftmaxParameter_Engine_DEFAULT) {
    engine = SoftmaxParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = SoftmaxParameter_Engine_CUDNN;
#endif
  }
  if (engine == SoftmaxParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new SoftmaxLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == SoftmaxParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNSoftmaxLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer);

// Get tanh layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetTanHLayer(const LayerParameter& param) {
  TanHParameter_Engine engine = param.tanh_param().engine();
  if (engine == TanHParameter_Engine_DEFAULT) {
    engine = TanHParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = TanHParameter_Engine_CUDNN;
#endif
  }
  if (engine == TanHParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new TanHLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == TanHParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNTanHLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(TanH, GetTanHLayer);

#ifdef WITH_PYTHON_LAYER
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetPythonLayer(const LayerParameter& param) {
  Py_Initialize();
  try {
    bp::object module = bp::import(param.python_param().module().c_str());
    bp::object layer = module.attr(param.python_param().layer().c_str())(param);
    return bp::extract<shared_ptr<PythonLayer<Dtype> > >(layer)();
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
}

REGISTER_LAYER_CREATOR(Python, GetPythonLayer);
#endif

// Layers that use their constructor as their default creator should be
// registered in their corresponding cpp files. Do not register them here.
}  // namespace caffe
