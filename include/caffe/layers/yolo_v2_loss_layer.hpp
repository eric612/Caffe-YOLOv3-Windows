#ifndef YOLOV2LOSSLAYER_H
#define YOLOV2LOSSLAYER_H

#include <vector>
#include <google/protobuf/repeated_field.h>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
    template <typename Dtype>
    Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2);
    template <typename Dtype>
    Dtype Calc_iou(const std::vector<Dtype>& box, const std::vector<Dtype>& truth);
    template <typename Dtype>
    Dtype Calc_rmse(const std::vector<Dtype>& box, const std::vector<Dtype>& truth);

    template<typename Dtype>
    class YoloV2LossLayer: public LossLayer<Dtype> {
    public:
        explicit YoloV2LossLayer(const LayerParameter& param)
            : LossLayer<Dtype>(param), diff_() {}

        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
              const vector<Blob<Dtype>*>& top);

        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
              const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "YoloV2Loss"; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
              const vector<Blob<Dtype>*>& top);
/*	    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
              const vector<Blob<Dtype>*>& top);*/
        
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
/*	    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);*/

        vector<float> biases_;
        int seen;
        int side_;
        int num_classes_;
        int num_boxes_;
        float box_scale_;
        float class_scale_;
        float object_scale_;
        float noobject_scale_;
        bool rescore_;
        bool constraint_;
        float thresh_;
        
        Blob<Dtype> diff_;
    };
}

#endif // YOLOV2LOSSLAYER_H
