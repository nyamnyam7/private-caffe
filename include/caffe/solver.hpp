#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"

#define DSP_LOSS
#ifdef DSP_LOSS
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#endif
namespace caffe {

/**
  * @brief Enumeration of actions that a client of the Solver may request by
  * implementing the Solver's action request function, which a
  * a client may optionally provide in order to request early termination
  * or saving a snapshot without exiting. In the executable caffe, this
  * mechanism is used to allow the snapshot to be saved when stopping
  * execution with a SIGINT (Ctrl-C).
  */
  namespace SolverAction {
    enum Enum {
      NONE = 0,  // Take no special action.
      STOP = 1,  // Stop training. snapshot_after_train controls whether a
                 // snapshot is created.
      SNAPSHOT = 2  // Take a snapshot, and keep training.
    };
  }

/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param,
      const Solver* root_solver = NULL);
  explicit Solver(const string& param_file, const Solver* root_solver = NULL);
  void Init(const SolverParameter& param);
  void InitTrainNet();
  void InitTestNets();

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  void SetActionFunction(ActionCallback func);
  SolverAction::Enum GetRequestedAction();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  void Restore(const char* resume_file);
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
#ifdef DSP_LOSS
  virtual ~Solver() { dsp_thread_run = 0; close_log(); }
  void init_test_net();
  inline shared_ptr<Net<Dtype> > test_net() { 
    //CHECK_NOTNULL(test_nets_[0].get())->ShareTrainedLayersWith(net_.get());
    return test_nets_[0]; 
  }
  void write_telog2(float v1, float v2);
  void set_train_label(const char* name);
  void set_test_label(const char* name);
  void copy_log_to(const char* dest);
  void copy_log_from(const char* src);
#else
  virtual ~Solver() {}
#endif
  inline const SolverParameter& param() const { return param_; }
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() { return iter_; }
  int current_step() { return current_step_; }
  void set_iter(int iter) { iter_ = iter; }
  void set_max_iter(int max_iter) { param_.set_max_iter(max_iter); }
  void set_current_step(int step) { current_step_ = step;  }

  // Invoked at specific points during an iteration
  class Callback {
   protected:
    virtual void on_start() = 0;
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  const vector<Callback*>& callbacks() const { return callbacks_; }
  void add_callback(Callback* value) {
    callbacks_.push_back(value);
  }

  void CheckSnapshotWritePermissions();
  /**
   * @brief Returns the solver type.
   */
  virtual inline const char* type() const { return ""; }

 protected:
  // Make and apply the update value for the current iteration.
  virtual void ApplyUpdate() = 0;
  string SnapshotFilename(const string extension);
  string SnapshotToBinaryProto();
  string SnapshotToHDF5();
  // The test routine
  void TestAll();
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  virtual void ResetSolver() {};  
  void DisplayOutputBlobs(const int net_id);
  void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);

  SolverParameter param_;
  int iter_;
  int current_step_;
  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;
  vector<Callback*> callbacks_;
  vector<Dtype> losses_;
  Dtype smoothed_loss_;

  // The root solver that holds root nets (actually containing shared layers)
  // in data parallelism
  const Solver* const root_solver_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;

#ifdef DSP_LOSS
  #define DSP_LOSS_VER "ver1.0.0"
  #define stack_cnt 10
  #define canv_loss_w 1200
  #define canv_loss_h 520
  #define canv_loss_plot_w 800
  #define canv_loss_plot_h 400
  #define canv_loss_plot_l 70
  #define canv_loss_plot_t 70
  #define canv_loss_plot_r (canv_loss_plot_l + canv_loss_plot_w)
  #define canv_loss_plot_b (canv_loss_plot_t + canv_loss_plot_h)
  #define canv_debug_w 1200
  #define canv_debug_h 800
  typedef struct {
	int iter, lcnt;
	Dtype loss[stack_cnt];	
  } LSTACK;
  vector<LSTACK> trloss_stack, teloss_stack;
  char loss_log[512], loss_jpg[512], debug_jpg[512];  
  int loss_mask[stack_cnt * 2];
  int plot_mask[10];
  cv::Mat canv_loss, canv_debug;  
  int dsp_thread_run;
  void dsp_loss_thread();
  void draw_all();
  int refresh;
  clock_t iter_tick0, iter_tick1;
  Dtype cur_lr;
  FILE* fp_log;
  char train_label[stack_cnt][256], test_label[stack_cnt][256];
  void init_log(const char* loss_log);
  void open_log();
  void close_log();
#endif
#if (defined(FORWARD_CHECK) || defined(BACKWARD_CHECK))
  NetParameter net_param_backup;
#endif
  DISABLE_COPY_AND_ASSIGN(Solver);
};

/**
 * @brief Solver that only computes gradients, used as worker
 *        for multi-GPU training.
 */
template <typename Dtype>
class WorkerSolver : public Solver<Dtype> {
 public:
  explicit WorkerSolver(const SolverParameter& param,
      const Solver<Dtype>* root_solver = NULL)
      : Solver<Dtype>(param, root_solver) {}

 protected:
  void ApplyUpdate() {}
  void SnapshotSolverState(const string& model_filename) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
  void RestoreSolverStateFromBinaryProto(const string& state_file) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
  void RestoreSolverStateFromHDF5(const string& state_file) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
