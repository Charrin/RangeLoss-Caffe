#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/uniform_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
UniformDataLayer<Dtype>::~UniformDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void UniformDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();

  const int label_size = this->layer_param_.image_data_param().label_size();

  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  /*string line;
  size_t pos;
  int label;
  while (std::getline(infile, line)) {
    pos = line.find_last_of(‘ ‘);
    label = atoi(line.substr(pos + 1).c_str());
    lines_.push_back(std::make_pair(line.substr(0, pos), label));
  }*/

  string line;
  size_t pos;
  std::string filename;
  int label_num = 0;
  while (std::getline(infile, line)) {
      std::vector<float> labels;
      float label;
      std::istringstream iss(line);
      iss >> filename;
      while (iss >> label){
          labels.push_back(label);
      }
      lines_.push_back(std::make_pair(filename, labels));
      if (labels[0] > label_num)
          label_num = labels[0];
  }

  CHECK(!lines_.empty()) << "File is empty";

  use_label_.resize(label_num + 1);
  use_data_.resize(label_num + 1);
  use_id_.resize(label_num + 1);

  for (int i = 0; i < label_num + 1; ++i) {
    use_label_[i] = i;
    use_id_[i]=0;
  }

  // add labels and data
  for (int i = 0; i < lines_.size(); ++i) {
    int l = lines_[i].second[0];
    use_data_[l].push_back(lines_[i]);
  }


  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleUniforms();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  /*
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  */
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  /*vector<int> label_shape(1, batch_size);*/
  //vector<int> label_shape(batch_size, label_size);
  vector<int> label_shape(2, batch_size);
  label_shape[1] = label_size;

  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void UniformDataLayer<Dtype>::ShuffleUniforms() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  //shuffle(lines_.begin(), lines_.end(), prefetch_rng);
  shuffle(use_label_.begin(), use_label_.end(), prefetch_rng);
}

template <typename Dtype>
void UniformDataLayer<Dtype>::ShuffleUniforms_data(int i) {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  //shuffle(lines_.begin(), lines_.end(), prefetch_rng);
  shuffle(use_data_[i].begin(), use_data_[i].end(), prefetch_rng);
  use_id_[i] = 0;
}

// This function is called on prefetch thread
template <typename Dtype>
void UniformDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();

  const int label_size = image_data_param.label_size();

  const int uniform_num = image_data_param.uniform_num();

  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  int l = use_label_[lines_id_];
  cv::Mat cv_img = ReadImageToCVMat(root_folder + use_data_[l][use_id_[l]].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << use_data_[l][use_id_[l]].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(use_label_.size(), lines_id_);
    l = use_label_[lines_id_];
    cv::Mat cv_img = ReadImageToCVMat(root_folder + use_data_[l][use_id_[l]].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << use_data_[l][use_id_[l]].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    /*prefetch_label[item_id] = lines_[lines_id_].second;*/
    //CHECK_EQ(label_size, lines_[lines_id_].second.size()) <<
    CHECK_EQ(label_size, use_data_[l][use_id_[l]].second.size()) <<
        "The input label size is not match the prototxt setting";
    for (int label_id = 0; label_id < label_size; ++label_id){
        prefetch_label[item_id*label_size + label_id] = use_data_[l][use_id_[l]].second[label_id];
    }

    DLOG(INFO) << "Load " << root_folder + use_data_[l][use_id_[l]].first << " label " << use_data_[l][use_id_[l]].second[0];


    // go to the next iter 
    use_id_[l]++;
    if (use_id_[l] >= use_data_[l].size()) {
        if (this->layer_param_.image_data_param().shuffle()) {
            ShuffleUniforms_data(l);
        }
    }
    if (item_id%uniform_num == uniform_num-1) {
        lines_id_++;
        if (lines_id_ >= use_label_.size()) {
            // We have reached the end. Restart from the first.
            DLOG(INFO) << "Restarting data prefetching from start.";
            lines_id_ = 0;
            if (this->layer_param_.image_data_param().shuffle()) {
            ShuffleUniforms();
            }
         }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(UniformDataLayer);
REGISTER_LAYER_CLASS(UniformData);

}  // namespace caffe
#endif  // USE_OPENCV
