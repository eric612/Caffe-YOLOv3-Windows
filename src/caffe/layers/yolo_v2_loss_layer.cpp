#include "caffe/layers/yolo_v2_loss_layer.hpp"
#include <caffe/blob.hpp>

#define SEEN_NUMBER 12800

namespace caffe {
//
    template <typename Dtype>
    struct Box
    {
        Dtype x, y, w, h,label;
    };

    template <typename Dtype>
    float logistic_activate(Dtype x)
    {
        return 1. / (1 + exp(-x));
    }

    template <typename Dtype>
    float logistic_gradient(Dtype x)
    {
        return (1-x) * x;
    }

    template <typename Dtype>
    Box<Dtype> dtype_to_box(const Dtype *dtype)
    {
        Box<Dtype> box;
        box.x = dtype[0];
        box.y = dtype[1];
        box.w = dtype[2];
        box.h = dtype[3];
		box.label = dtype[4];
        return box;
    }

    template <typename Dtype>
    Box<Dtype> get_region_box(const Dtype *x, vector<float>& biases, int n, int SIZE, int i, int j, int side){
        Box<Dtype> box;
        box.x = (i + x[0 * SIZE]) / side;
        box.y = (j + x[1 * SIZE]) / side;
        box.w = exp(x[2 * SIZE]) * biases[2 * n];
        box.h = exp(x[3 * SIZE]) * biases[2 * n + 1];
        return box;
    }

    template <typename Dtype>
    Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
      Dtype left = std::max(x1 - w1 / 2, x2 - w2 / 2);
      Dtype right = std::min(x1 + w1 / 2, x2 + w2 / 2);
      return right - left;
    }

    template <typename Dtype>
    Dtype Calc_iou(const Box<Dtype> box, const Box<Dtype> truth) {
      Dtype w = Overlap(box.x, box.w, truth.x, truth.w);
      Dtype h = Overlap(box.y, box.h, truth.y, truth.h);
      if (w < 0 || h < 0) return 0;
      Dtype inter_area = w * h;
      Dtype union_area = box.w * box.h + truth.w * truth.h - inter_area;
      return inter_area / union_area;
    }

    template <typename Dtype>
    Dtype abs(Dtype x)
    {
        if(x < 0)
            return -x;
        return x;
    }

    template <typename Dtype>
    Dtype Calc_rmse(const Box<Dtype>& truth, const Box<Dtype>& box, Dtype &coord_loss, Dtype &area_loss, float scale) {
        float coord_ = scale * (abs(box.x-truth.x) + abs(box.y-truth.y));
        float area_  = scale * (abs(box.w-truth.w) + abs(box.h-truth.h));
        coord_loss += coord_;
        area_loss  += area_;
        return (coord_ + area_);
    }

    template <typename Dtype>
    float delta_region_box(const Box<Dtype>& truth_box, const Box<Dtype>& pred_box, Dtype* diff, int index, float scale, int SIZE)
    {
        float iou = Calc_iou(truth_box, pred_box);

        diff[index + 0 * SIZE] = scale * (pred_box.x - truth_box.x)
            * logistic_gradient(pred_box.x);
        diff[index + 1 * SIZE] = scale * (pred_box.y - truth_box.y)
            * logistic_gradient(pred_box.y);

        diff[index + 2 * SIZE] = scale * (pred_box.w - truth_box.w) * pred_box.w;
        diff[index + 3 * SIZE] = scale * (pred_box.h - truth_box.h) * pred_box.h;

        return iou;
    }

    template <typename Dtype>
    float delta_region_box(Box<Dtype>& truth_box, Box<Dtype>& pred_box, 
            std::vector<float>& biases, int n, int side, Dtype* diff, 
            int index, float scale) {
        int SHIFT = side * side;
        diff[index + 0 * SHIFT] = scale * (pred_box.x - truth_box.x);
        diff[index + 1 * SHIFT] = scale * (pred_box.y - truth_box.y);

/**/
        diff[index + 2 * SHIFT] = scale * (pred_box.w - truth_box.w) * pred_box.w;
        diff[index + 3 * SHIFT] = scale * (pred_box.h - truth_box.h) * pred_box.h;
/** /
        diff[index + 2 * SHIFT] = scale * (log(pred_box.w / biases[2 * n + 0])
                     - log(truth_box.w / biases[2 * n + 0]));
        diff[index + 3 * SHIFT] = scale * (log(pred_box.h / biases[2 * n + 1])
                     - log(truth_box.h / biases[2 * n + 1])); 
/* **/

        return Calc_iou(truth_box, pred_box);
    }

    template <typename Dtype>
    void YoloV2LossLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::LayerSetUp(bottom, top);

        YoloV2LossParameter param = this->layer_param().yolo_v2_loss_param();
        const google::protobuf::RepeatedField<float> &tmpBiases_ = param.biaes();
        for(google::protobuf::RepeatedField<float>::const_iterator 
            iterator = tmpBiases_.begin(); iterator != tmpBiases_.end(); ++iterator)
        {
            biases_.push_back(*iterator);
            //LOG(INFO) << *iterator;
        }

        side_           = param.side();
        CHECK_GE(side_, 0) << "side size must bigger then 0";
        num_classes_    = param.num_classes();
        CHECK_GE(num_classes_, 0) << "class number must bigger then 0";
        num_boxes_      = param.num_object();
        CHECK_GE(num_boxes_, 0) << "box number must bigger then 0";
        CHECK_EQ(biases_.size() , num_boxes_ * 2) << "biases size and num_boxes doesn't match";
        seen            = 0;
        box_scale_      = param.box_scale();
        class_scale_    = param.class_scale();
        object_scale_   = param.object_scale();
        noobject_scale_ = param.noobject_scale();
        rescore_        = param.rescore();
        constraint_     = param.constraint();
        thresh_         = param.thresh();

        int input_count = bottom[0]->count(1);
        int label_count = bottom[1]->count(1);

        int tmp_input_count = side_ * side_ * (num_classes_ + (1 + 4)) * num_boxes_;
        int tmp_label_count = 150;
        CHECK_EQ(input_count, tmp_input_count);
        CHECK_EQ(label_count, tmp_label_count);
    }

    template <typename Dtype>
    void YoloV2LossLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::Reshape(bottom, top);
        diff_.ReshapeLike(*bottom[0]);
    }


    template <typename Dtype>
    void YoloV2LossLayer<Dtype>::Forward_cpu(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        Dtype max_diff = 0;
        Dtype min_diff = 999;
        const Dtype* input_data = bottom[0]->cpu_data();
        const Dtype* label_data = bottom[1]->cpu_data();
        const int pow_side = side_ * side_;
        const int size = pow_side * (5 + num_classes_);
        int recall = 0;

        Dtype* diff = diff_.mutable_cpu_data();
        Dtype class_loss(0.0), noobj_score(0.0), obj_score(0.0), avg_best_iou(0.0);
        int obj_count = 0;

        int locations = pow(side_, 2);

        caffe_set(diff_.count(), Dtype(0.), diff);

        for(int batch = 0; batch < bottom[0]->num(); ++batch)
        {
            int truth_index =  batch * bottom[1]->count(1);
			std::vector<caffe::Box<Dtype> > true_box_list;
			true_box_list.clear();
			//int b = bottom[0]->num();
			for (int t = 0; t < 30; ++t) {
				Dtype x = label_data[batch * 30 * 5 + t * 5 + 0];
				if (!x) break;
				true_box_list.push_back(dtype_to_box(&label_data[batch * 30 * 5 + t * 5 ]));

			}
            //for(int j = 0; j < locations; ++j){
            //   bool isobj = label_data[locations + truth_index + j];
            //   if(!isobj) continue;
            //   true_box_list.push_back(dtype_to_box(label_data + truth_index + locations * 3 + j * 4));
            //}
            for(int j = 0; j < locations; ++j){
                for(int n = 0; n < num_boxes_; ++n){
                    int boxes_index = j + size * n + batch * bottom[0]->count(1);
                    Box<Dtype> pred_box = get_region_box(input_data + boxes_index, 
                        biases_, n, pow_side, j % side_, j / side_, side_);
                    float best_iou = 0;
                    for(int index = 0; index < true_box_list.size(); ++index) {
                        float iou = Calc_iou(true_box_list[index], pred_box);
                        if(iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    int obj_index = boxes_index + 4 * pow_side;
                    noobj_score += abs(input_data[obj_index]);
                    diff[obj_index] = noobject_scale_ * (input_data[obj_index] - 0.);
                    if(best_iou > thresh_) {
                        diff[obj_index] = 0;
                    }
                    if(diff[obj_index] > max_diff) 
                        max_diff = diff[obj_index];
                    if(diff[obj_index] < min_diff) 
                        min_diff = diff[obj_index];
                    if(seen < SEEN_NUMBER) {
                        Box<Dtype> truth;
                        truth.x = ((j % side_) + .5) / side_;
                        truth.y = ((j / side_) + .5) / side_;
                        truth.w = biases_[2 * n + 0];
                        truth.h = biases_[2 * n + 1];
                        delta_region_box(truth, pred_box, biases_, n, side_, diff,
                            boxes_index, .01);
                    }
                }
            }
            for(int index = 0; index < true_box_list.size(); ++index){
                float best_iou = 0;
                int best_n = 0;
                Box<Dtype> shift_box = true_box_list[index];
                shift_box.x = shift_box.y = 0;
                int w = true_box_list[index].x * side_;
                int h = true_box_list[index].y * side_;
                for(int n = 0; n < num_boxes_; ++n) {
                    int box_index = (h * side_ + w) + size * n + batch * bottom[0]->count(1);
                    Box<Dtype> pred_box = get_region_box(input_data + box_index, 
                        biases_, n, pow_side, w, h, side_);
                    pred_box.x = pred_box.y = 0;
                    bool bias_match = true; //TODO
                    if(bias_match) {
                        pred_box.w = biases_[2 * n + 0];
                        pred_box.h = biases_[2 * n + 1];
                    }
                    float iou = Calc_iou(shift_box, pred_box);
                    if(iou > best_iou) {
                        best_iou = iou;
                        best_n = n;
                    }
                }
                int box_index = (h * side_ + w) + size * best_n + batch * bottom[0]->count(1);
                Box<Dtype> pred_box = get_region_box(input_data + box_index, 
                        biases_, best_n, pow_side, w, h, side_);

                float iou = delta_region_box(true_box_list[index], pred_box,biases_,
                                best_n, side_, diff, box_index, 
                                    box_scale_ * (2 - true_box_list[index].x *
                                            true_box_list[index].y));
/*
                LOG(INFO) << "index : " << box_index << " "  <<
                             "obj_score: " << input_data[box_index + 4 * pow_side] <<
                             " box_index: " << box_index << " " <<
                             "true x: " << true_box_list[index].x << " pred x: " << pred_box.x << " " <<
                             "true y: " << true_box_list[index].y << " pred y: " << pred_box.y << " " <<
                             "true w: " << true_box_list[index].w << " pred w: " << pred_box.w << " " << 
                             "true h: " << true_box_list[index].h << " pred h: " << pred_box.h << " " <<
                             "first score :" << input_data[0]; 
*/
                if(iou > .5) ++recall;
                avg_best_iou += iou;
                
                int obj_index = box_index + 4 * pow_side;
                obj_score += input_data[obj_index];
                diff[obj_index] = object_scale_ * (input_data[obj_index] - 1);
                if (rescore_) {
                    diff[obj_index] = object_scale_ * (input_data[obj_index] - iou);
                }
        
                    if(diff[obj_index] > max_diff) 
                        max_diff = diff[obj_index];
                    if(diff[obj_index] < min_diff) 
                        min_diff = diff[obj_index];
                //int label = static_cast<int>(label_data[locations * 2 + truth_index + h * side_ + w]);
				int label = true_box_list[index].label;
				//int class_label = label_data[t * 5 + b * 30 * 5 + 0];
                for(int index = 0; index < num_classes_; ++index) {
                    int class_index = obj_index + pow_side * (index + 1);
                    Dtype target(index == label);
                    diff[class_index] = class_scale_ * (input_data[class_index] - target);
                    if(target) {
                        class_loss += class_scale_ * abs(input_data[class_index] - target);
                    }
                }
            }
            obj_count += true_box_list.size();
        }
        avg_best_iou /= obj_count;
        class_loss /= obj_count;
        obj_score /= obj_count;
        int diff_size = bottom[0]->num() * pow_side * num_boxes_;
        noobj_score /= (diff_size - obj_count);

        LOG(INFO) << "class_loss: " << class_loss << " obj_score: " << obj_score
                  << " noobj_score: " << noobj_score << " avg_best_iou " << avg_best_iou
                  << " Avg Recall: " << float(recall) / obj_count << " count: " << obj_count;

        top[0]->mutable_cpu_data()[0] = noobj_score * noobject_scale_ + (1 - obj_score) * object_scale_ + 
                class_loss + (1 - avg_best_iou) + (1 - float(recall) / obj_count);

        if(seen < SEEN_NUMBER) {
            //LOG(INFO) << "seen is: " << seen;
            seen += bottom[0]->num();
        }
    }

    template <typename Dtype>
    void YoloV2LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        if (propagate_down[1]) {
          LOG(FATAL) << this->type()
                     << " Layer cannot backpropagate to label inputs.";
        }
        if (propagate_down[0]) {
          const Dtype sign(1.);
          const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();
          //LOG(INFO) << "alpha: " << alpha;
          caffe_cpu_axpby(
              bottom[0]->count(),
              alpha,
              diff_.cpu_data(),
              Dtype(0),
              bottom[0]->mutable_cpu_diff());
        }
    }


#ifdef CPU_ONLY
    //STUB_GPU(YoloV2LossLayer);
#endif
   
    INSTANTIATE_CLASS(YoloV2LossLayer);
    //REGISTER_LAYER_CLASS(YoloV2Loss);
}

