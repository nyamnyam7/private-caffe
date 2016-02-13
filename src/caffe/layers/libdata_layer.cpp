#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <vector>
#include <dlfcn.h>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/libdata_layer.hpp"

namespace caffe {


template <typename Dtype>
LibDataLayer<Dtype>::LibDataLayer(const LayerParameter& param)
  : BaseExtendedPrefetchingDataLayer<Dtype>(param)
{
}
    
template <typename Dtype>
LibDataLayer<Dtype>::~LibDataLayer() {
  this->StopInternalThread();
  dlclose(libhandle_);
}


template <typename Dtype>
void LibDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {

  libpath_ = this->layer_param_.lib_external_param().libpath();
  libparam_ = this->layer_param_.lib_external_param().libparam();
  batch_size_ = this->layer_param_.lib_external_param().batch_size();
  pos_ = 0;

  CHECK (libpath_ != "");
      
  libhandle_ = dlopen(libpath_.c_str(), RTLD_LAZY);
  if (!libhandle_) fputs(dlerror(), stderr);
  CHECK(libhandle_);

  char* errstr;
  initfunc_ = reinterpret_cast<DataInitFunction>(dlsym(libhandle_, "init"));
  if ((errstr = dlerror()) != NULL) {
      fputs(errstr, stderr);
      CHECK(!errstr);
  }
  memset(&iface_, 0, sizeof(iface_));

  for (int i=0; i< this->PREFETCH_COUNT; i++)
    initfunc_(&iface_, &this->prefetch_[i].userdata_, libparam_);

  num_blobs_ = top.size();

  for (int i=0; i< this->PREFETCH_COUNT; ++i) {
    for (int j=0; j<num_blobs_; ++j) {
      std::vector<int> shape;
      iface_.get_shape(this->prefetch_[i].userdata_, j, batch_size_, shape);
      shape[0] = batch_size_;
      this->prefetch_[i].data_[j]->Reshape(shape);
      if (i==0) top[j]->Reshape(shape);
    }
  }
}


template<typename Dtype>
void LibDataLayer<Dtype>::load_batch(ExtendedBatch<Dtype>* batch) {
  CHECK(batch->data_[0]->count());
  size_t curpos;

  #pragma omp critical
  {
    curpos = pos_;
    pos_ += batch_size_;
  }

  iface_.reset(batch->userdata_, curpos); 
  for (int i=0; i<batch->data_.size(); i++) {
    Dtype* top_data = batch->data_[i]->mutable_cpu_data();
    iface_.fill(batch->userdata_, i, curpos, batch->data_[i]->shape(), top_data);
  }

}


INSTANTIATE_CLASS(LibDataLayer);
REGISTER_LAYER_CLASS(LibData);

}  // namespace caffe
