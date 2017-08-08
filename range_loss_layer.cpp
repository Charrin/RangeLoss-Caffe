#include <vector>

#include "caffe/layers/range_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RangeLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top) {
  K_ = bottom[0]->count() / bottom[0]->num();
  choose_k_ = this->layer_param_.range_loss_param().choose_k();
  inter_weight_ = this->layer_param_.range_loss_param().inter_weight();
  intra_weight_ = this->layer_param_.range_loss_param().intra_weight();
  margin_ = this->layer_param_.range_loss_param().margin();
}

template <typename Dtype>
void RangeLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  vector<int> shape(2);
  shape[0]=1; shape[1]=K_;
  distance_.Reshape(shape);
}

template <typename Dtype>
void RangeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LOG(INFO) << "bottom data " << bottom[0]->asum_data() << " avg " << bottom[0]->asum_data() / bottom[0]->count();
  std::map<int, std::vector<int> >::iterator l_it;
  // clear
  inter_distance.clear();
  for (int i = 0; i < intra_distance.size(); ++i) {
    intra_distance[i].clear();
  }
  intra_distance.clear();
  for (l_it = map_class.begin(); l_it != map_class.end(); l_it++) {
    l_it->second.clear();
  }
  map_class.clear();
  class_label_.clear();


  int count = bottom[0]->count();
  M_ = bottom[0]->num(); 
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data(); 
  Dtype* distance_data = distance_.mutable_cpu_data();

  // add features index
  //
  // the i-th distance_data 
  for (int i = 0; i < M_; i++) { 
      const int label_value = static_cast<int>(label[i]);  
      l_it=map_class.find(label_value);
      if(l_it==map_class.end()) {
        std::vector<int> feat_index;
        feat_index.push_back(i);
        map_class[label_value] = feat_index;
        class_label_.push_back(label_value);
      } else {
        l_it->second.push_back(i);
      }
  }

  // calculate intra class distance
  //
  intra_distance.resize(class_label_.size()); 
  std::vector<int> shape(2);
  shape[0] = class_label_.size();
  shape[1] = K_;
  center_.Reshape(shape);
  shape[1] = 1;
  S_.Reshape(shape);

  Dtype* center_data = center_.mutable_cpu_data();
  Dtype* S_data = S_.mutable_cpu_data();

  caffe_set(center_.count(), Dtype(0), center_data);
  caffe_set(S_.count(), Dtype(0), S_data);

  loss_intra = 0;
  loss_inter = 0;

  for (int d = 0; d < class_label_.size(); ++d) {
      int class_ind = class_label_[d];
      DLOG(INFO) << "class ind " << class_ind << " d " << d;
      std::vector<int> feat_index = map_class[class_ind];
      intra_distance[d].resize(feat_index.size()*(feat_index.size()-1)/2);
      int ind = 0;
      for (int i = 0; i < feat_index.size(); ++i) {
        for (int j = i+1; j < feat_index.size(); ++j) {
            caffe_sub(K_, bottom_data + feat_index[i] * K_, bottom_data + feat_index[j] * K_, distance_data);
            intra_distance[d][ind].dist = caffe_cpu_dot(K_, distance_.cpu_data(), distance_.cpu_data());
            intra_distance[d][ind].ind1 = i;
            intra_distance[d][ind].ind2 = j;
            //LOG(INFO) << "feat index " << feat_index[i] << " " << feat_index[j] << " dist " << intra_distance[d][ind].dist;
            ind += 1;
        }
        // class center
        caffe_axpy(K_, Dtype(1), bottom_data + feat_index[i] * K_, center_data + d * K_);
      }
      // center 
      caffe_scal(K_, Dtype(1./feat_index.size()), center_data + d * K_);

      // intra loss
      std::sort(intra_distance[d].begin(),intra_distance[d].end(),std::greater<Dist>());
      Dtype loss_d = 0;
      for (int k = 0; k < choose_k_; ++k) {
          loss_d += Dtype(1/intra_distance[d][k].dist);
      }
      //LOG(INFO) << "class " << d << " 1st " << intra_distance[d][0].dist << " 2nd " << intra_distance[d][1].dist;
      S_data[0] = loss_d;
      loss_intra += Dtype(choose_k_/loss_d);
      S_data += 1;
  }

  // claculate inter loss
  //
  int d = 0;
  inter_distance.resize(class_label_.size() * (class_label_.size()-1) / 2);
  for (int i = 0; i < class_label_.size(); ++i) {
    for (int j = i+1; j < class_label_.size(); ++j) {
        caffe_sub(K_, center_data + i * K_, center_data + j * K_, distance_data);
        inter_distance[d].dist = caffe_cpu_dot(K_, distance_.cpu_data(), distance_.cpu_data()); 
        inter_distance[d].ind1 = i; 
        inter_distance[d].ind2 = j; 
        d+=1;
    }
  }

  std::sort(inter_distance.begin(),inter_distance.end(),std::less<Dist>()); 
  LOG(INFO) << " inter_distance " << inter_distance[0].dist;
  loss_inter = std::max(margin_ - inter_distance[0].dist, Dtype(0));

  top[0]->mutable_cpu_data()[0] = inter_weight_ * loss_inter + intra_weight_ * loss_intra;
  LOG(INFO) << "loss_intra " << loss_intra << " loss_inter " << loss_inter << " loss " << top[0]->mutable_cpu_data()[0];
}

template <typename Dtype>
void RangeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 1; ++i) {
    if (propagate_down[i]) {
		Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
		const Dtype* bottom_data = bottom[i]->cpu_data();
		const Dtype* center_data = center_.cpu_data();
        const Dtype* S_data = S_.cpu_data();
		Dtype* distance_data = distance_.mutable_cpu_data();

		caffe_set(bottom[i]->count(), Dtype(0), bottom_diff);

        std::map<int, std::vector<int> >::iterator l_it;
        // intra gradiant
        for (int d = 0; d < class_label_.size(); ++d) {
            int class_ind = class_label_[d];
       		std::vector<int> feat_index = map_class[class_ind];
        	for (int k = 0; k < choose_k_; ++k) {
				int ind1 = feat_index[intra_distance[d][k].ind1];
				int ind2 = feat_index[intra_distance[d][k].ind2];
				caffe_sub(K_, bottom_data + ind1 * K_, bottom_data + ind2 * K_, distance_data);
				//caffe_abs(K_, distance_data, distance_data);
				Dtype g = 2 * choose_k_ / ((intra_distance[d][k].dist * S_data[0])*(intra_distance[d][k].dist * S_data[0]));
                //LOG(INFO) << "ind " << ind1 << " " << ind2 << " distance_ " << distance_.asum_data() << " distance_ avg " << distance_.asum_data() / distance_.count() << " g " << g;
				caffe_cpu_axpby(K_, g * intra_weight_, distance_data, Dtype(1.), bottom_diff + ind1 * K_);
				caffe_cpu_axpby(K_, -(g * intra_weight_), distance_data, Dtype(1.), bottom_diff + ind2 * K_);
			}
            S_data += 1;
        }

        //LOG(INFO) << "after intra, diff asum " << bottom[i]->asum_diff() << " avg " << bottom[i]->asum_diff() / bottom[i]->count();

        if (loss_inter > 0) {
		    // inter gradiant
            // class
		    int class1 = class_label_[inter_distance[0].ind1];
		    int class2 = class_label_[inter_distance[0].ind2];

		    caffe_sub(K_, center_data + inter_distance[0].ind1 * K_, center_data + inter_distance[0].ind2 * K_, distance_data);
            //LOG(INFO) << "index " << inter_distance[0].ind1 << " " << inter_distance[0].ind2;
            //LOG(INFO) << inter_distance[0].dist << " " << inter_distance[1].dist << " " << inter_distance[2].dist << " " << inter_distance[3].dist;
		    //caffe_abs(K_, distance_data, distance_data);

            // class 1
            for (int c = 0; c < map_class[class1].size(); ++c) {
                int ind = map_class[class1][c];
		        caffe_cpu_axpby(K_, (-inter_weight_)/(2*map_class[class2].size()), distance_data, Dtype(1.), bottom_diff + ind * K_);
            }

            // class2
            for (int c = 0; c < map_class[class2].size(); ++c) {
                int ind = map_class[class2][c];
		        caffe_cpu_axpby(K_, inter_weight_/(2*map_class[class1].size()), distance_data, Dtype(1.), bottom_diff + ind * K_);
            }
            //LOG(INFO) << "after inter, diff asum " << bottom[i]->asum_diff() << " avg " << bottom[i]->asum_diff() / bottom[i]->count();
        }

        caffe_scal(bottom[i]->count(), top[0]->cpu_diff()[0], bottom_diff);
    }  

    LOG(INFO) << "diff asum " << bottom[i]->asum_diff() << " avg " << bottom[i]->asum_diff() / bottom[i]->count();

  }
}

#ifdef CPU_ONLY
STUB_GPU(RangeLossLayer);
#endif

INSTANTIATE_CLASS(RangeLossLayer);
REGISTER_LAYER_CLASS(RangeLoss);

}  // namespace caffe
