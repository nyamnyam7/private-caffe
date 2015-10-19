#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <vector>
#include <dlfcn.h>

#include "caffe/lib_external.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {


template <typename Dtype>
LibProcessLayer<Dtype>::~LibProcessLayer() {
  dlclose(libhandle_);
}


template <typename Dtype>
void LibProcessLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
   const vector<Blob<Dtype>*>& top) {

  libpath_ = this->layer_param_.lib_external_param().libpath();
  libparam_ = this->layer_param_.lib_external_param().libparam();

  libhandle_ = dlopen(libpath_.c_str(), RTLD_LAZY);
  if (!libhandle_) fputs(dlerror(), stderr);
  CHECK(libhandle_);

  char* errstr;
  initfunc_ = reinterpret_cast<ProcessInitFunction>(dlsym(libhandle_, "init"));
  if ((errstr = dlerror()) != NULL) {
    fputs(errstr, stderr);
    CHECK(!errstr);
  }
  memset(&iface_, 0, sizeof(iface_));
  libuserdata_ = NULL;
  initfunc_(&iface_, &libuserdata_, libparam_);
}

template <typename Dtype>
void LibProcessLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
   const vector<Blob<Dtype>*>& top) {

  std::vector< std::vector <int> > shape;
  for (int i=0; i<bottom.size(); i++)
    shape.push_back(bottom[i]->shape());
  iface_.set_bottom_shape(libuserdata_, shape);
  
  shape.clear();
  iface_.get_top_shape(libuserdata_, shape);

  CHECK(shape.size() == top.size());
  for (int i=0; i<top.size(); i++)
    top[i]->Reshape(shape[i]);
}

template <typename Dtype>
void LibProcessLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  vector<const void *> data_bottom;
  vector<void *> data_top;

  for (int i=0; i<bottom.size(); i++)
    data_bottom.push_back(bottom[i]->cpu_data());

  for (int i=0; i<top.size(); i++)
    data_top.push_back(top[i]->mutable_cpu_data());

  iface_.forward_cpu(libuserdata_, data_bottom, data_top);
}

template <typename Dtype>
void LibProcessLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  vector<void *> data_bottom;
  vector<const void *> data_top;

  for (int i=0; i<bottom.size(); i++)
    data_bottom.push_back(bottom[i]->mutable_cpu_data());

  for (int i=0; i<top.size(); i++)
    data_top.push_back(top[i]->cpu_data());

  iface_.backward_cpu(libuserdata_, data_top, propagate_down, data_bottom);
}



template <typename Dtype>
void LibProcessLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  vector<const void *> data_bottom;
  vector<void *> data_top;

  if (iface_.forward_gpu) {
    for (int i=0; i<bottom.size(); i++)
      data_bottom.push_back(bottom[i]->gpu_data());

    for (int i=0; i<top.size(); i++)
      data_top.push_back(top[i]->mutable_gpu_data());
  
    iface_.forward_gpu(libuserdata_, data_bottom, data_top);
  }
  else Forward_cpu(bottom, top);
}

template <typename Dtype>
void LibProcessLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  vector<void *> data_bottom;
  vector<const void *> data_top;

  if (iface_.backward_gpu) {
    for (int i=0; i<bottom.size(); i++)
      data_bottom.push_back(bottom[i]->mutable_gpu_data());

    for (int i=0; i<top.size(); i++)
      data_top.push_back(top[i]->gpu_data());

    iface_.backward_gpu(libuserdata_, data_top, propagate_down, data_bottom);
  }
  else Backward_cpu(top, propagate_down, bottom);
}


INSTANTIATE_CLASS(LibProcessLayer);
REGISTER_LAYER_CLASS(LibProcess);

}  // namespace caffe
