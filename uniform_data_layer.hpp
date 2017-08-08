#ifndef CAFFE_UNIFORM_DATA_LAYER_HPP_
#define CAFFE_UNIFORM_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class UniformDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit UniformDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~UniformDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "UniformData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleUniforms();
  virtual void ShuffleUniforms_data(int i);
  virtual void load_batch(Batch<Dtype>* batch);

  //vector<std::pair<std::string, int> > lines_;
  vector<std::pair<std::string, std::vector<float> > > lines_;
  vector<int> use_label_;
  vector<vector<std::pair<std::string, std::vector<float> > > > use_data_;
  int lines_id_;
  vector<int> use_id_;
};


}  // namespace caffe

#endif  // CAFFE_UNIFORM_DATA_LAYER_HPP_
