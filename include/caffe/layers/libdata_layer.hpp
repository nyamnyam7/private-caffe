#ifndef CAFFE_LIBDATA_LAYER_HPP_
#define CAFFE_LIBDATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/lib_external.hpp"
#include "caffe/layers/base_data_layer.hpp"

namespace caffe {



/**
 * @brief Provides data to the Net from .so file.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class LibDataLayer : public BaseExtendedPrefetchingDataLayer<Dtype> {
 public:
  explicit LibDataLayer(const LayerParameter& param);
  virtual ~LibDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "LibData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 100; }

 protected:
  virtual void load_batch(ExtendedBatch<Dtype>* batch);

  size_t pos_;

  std::string libpath_;
  std::string libparam_;

  void* libhandle_;
  void* libuserdata_;
  DataInitFunction initfunc_;
  LibDataInterface iface_;

  int num_blobs_;
  int batch_size_;

};


}  // namespace caffe

#endif  // CAFFE_LIBDATA_LAYER_HPP_
