#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void OnehotLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  OnehotParameter onehot_param = this->layer_param_.onehot_param();

  CHECK(onehot_param.has_num_classes()) << "num_classes should be defined";
  num_classes_ = onehot_param.num_classes();
  
  CHECK_GT(num_classes_, 0) << "num_classes should be larger than 0";
    
}


template <typename Dtype>
void OnehotLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(1, bottom[0]->num_axes()) << "Input must have 1 axis.(Label layer)";
  num_ = bottom[0]->num();

  // we will initialize the top blob and the vector index part.
  std::vector<int> shape;
  shape.push_back(bottom[0]->num());
  shape.push_back(num_classes_);
  top[0]->Reshape(shape);
}


template <typename Dtype>
void OnehotLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i=0; i<num_; i++)
  {
    for (int j=0; j<num_classes_; j++)
    {
      top_data[i * num_classes_ + j] = (bottom_data[i] == j);
    }
  }
}

template <typename Dtype>
void OnehotLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;
}


#ifdef CPU_ONLY
STUB_GPU(OnehotLayer);
#endif

INSTANTIATE_CLASS(OnehotLayer);

}  // namespace caffe
