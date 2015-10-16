#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <vector>
#include <dlfcn.h>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {


template <typename Dtype>
LibDataLayer<Dtype>::LibDataLayer(const LayerParameter& param)
  : BaseExtendedPrefetchingDataLayer<Dtype>(param)
{
}
    
template <typename Dtype>
LibDataLayer<Dtype>::~LibDataLayer() {
  this->StopInternalThread();
}


template <typename Dtype>
void LibDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {

  libpath_ = this->layer_param_.lib_data_param().libpath();
  libparam_ = this->layer_param_.lib_data_param().libparam();
  batch_size_ = this->layer_param_.lib_data_param().batch_size();
  pos_ = 0;

  libhandle_ = dlopen(libpath_.c_str(), RTLD_LAZY);
  if (!libhandle_) fputs(dlerror(), stderr);
  CHECK(libhandle_);

  char* errstr;
  initfunc_ = reinterpret_cast<InitFunction>(dlsym(libhandle_, "init"));
  if ((errstr = dlerror()) != NULL) {
      fputs(errstr, stderr);
      CHECK(!errstr);
  }
  memset(&iface_, 0, sizeof(iface_));
  libuserdata_ = NULL;
  initfunc_(&iface_, &libuserdata_, libparam_);

  num_blobs_ = top.size();

  for (int i=0; i<num_blobs_; i++) {
    std::vector<int> shape;
    iface_.get_shape(libuserdata_, i, batch_size_, shape);
    shape[0] = batch_size_;
    top[i]->Reshape(shape);
    for (int j=0; j< this->PREFETCH_COUNT; ++j)
      this->prefetch_[j].data_[i]->Reshape(shape);
  }
}


template<typename Dtype>
void LibDataLayer<Dtype>::load_batch(ExtendedBatch<Dtype>* batch) {
  CHECK(batch->data_[0]->count());
  iface_.reset(libuserdata_, pos_); 
  for (int i=0; i<batch->data_.size(); i++) {
    Dtype* top_data = batch->data_[i]->mutable_cpu_data();
    iface_.fill(libuserdata_, i, pos_, batch->data_[i]->shape(), top_data);
  }
  pos_ += batch_size_;
}


INSTANTIATE_CLASS(LibDataLayer);
REGISTER_LAYER_CLASS(LibData);

}  // namespace caffe
