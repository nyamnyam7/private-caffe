#ifndef CAFFE_EXTERNAL_DATA_HPP_
#define CAFFE_EXTERNAL_DATA_HPP_

#include <string>
#include <vector>

namespace caffe {

struct LibDataInterface {
    void (*get_shape)(void* mem, int blobidx, std::vector<int>& shape);
    void (*reset)(void* mem, size_t pos, size_t batchsz);
    void (*fill)(void* mem, int blobidx, size_t pos, const std::vector<int>& shape, void* data);
    int  (*num_top_blobs)(void* mem);
};

// returns a library specific structure
typedef void(*InitFunction)(LibDataInterface* iface, void** mem, const std::string& userstring);

};

#endif
