#ifndef CAFFE_EXTERNAL_DATA_HPP_
#define CAFFE_EXTERNAL_DATA_HPP_

#include <string>
#include <vector>

namespace caffe {

struct LibDataInterface {
    void (*get_shape)(void* mem, int blobidx, size_t batchsz, std::vector<int>& shape);
    void (*reset)(void* mem, size_t pos);
    void (*fill)(void* mem, int blobidx, size_t pos, const std::vector<int>& shape, void* data);
    int  (*num_top_blobs)(void* mem);
};

struct LibProcessInterface {
    void (*set_bottom_shape)(void* mem, const std::vector< std::vector<int> >& shape);
    void (*get_top_shape)(void* mem, std::vector< std::vector<int> >& shape);
    void (*forward_cpu)(void* mem, const std::vector<const void*>& bottom, const std::vector<void*>& top);
    void (*forward_gpu)(void* mem, const std::vector<const void*>& bottom, const std::vector<void*>& top);
    void (*backward_cpu)(void* mem, const std::vector<const void*>& top, const std::vector<bool>& propagate_down, const std::vector<void*>& bottom);
    void (*backward_gpu)(void* mem, const std::vector<const void*>& top, const std::vector<bool>& propagate_down, const std::vector<void*>& bottom);
};

// returns a library specific structure
typedef void(*DataInitFunction)(LibDataInterface* iface, void** mem, const std::string& userstring);
typedef void(*ProcessInitFunction)(LibProcessInterface* iface, void** mem, const std::string& userstring);

};

#endif
