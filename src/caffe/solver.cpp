#include <cstdio>

#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(param_file, &param);
  Init(param);
}

static void create_dir_recursive(const char* path, bool full) {
  for (int i = 0; i < strlen(path); i++) {
    if (path[i] == '/' || path[i] == '\\') {
      char path0[512] = { 0, };
      strcpy(path0, path);
      path0[i] = '\0';
      boost::filesystem::create_directory(boost::filesystem::path(path0));
    }
  }
  if (full) boost::filesystem::create_directory(boost::filesystem::path(path));
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  CHECK(Caffe::root_solver() || root_solver_)
      << "root_solver_ needs to be set for all non-root solvers";
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
    << std::endl << param.DebugString();
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  if (strlen(param_.snapshot_prefix().c_str()) != 0) {
    create_dir_recursive(param_.snapshot_prefix().c_str(), false);
  }
  CheckSnapshotWritePermissions();
  if (Caffe::root_solver() && param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();
  if (Caffe::root_solver()) {
    InitTestNets();
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  current_step_ = 0;
#ifdef DSP_LOSS
  if (strlen(param_.log_path().c_str()) != 0) {
	const char* prototxt = (strlen(param_.net().c_str()) != 0) ? param_.net().c_str() : (strlen(param_.train_net().c_str()) != 0) ? param_.train_net().c_str() : "caffe";
    int i = strlen(prototxt);
	for (; i >= 0; i--) if (prototxt[i] == '/' || prototxt[i] == '\\') break;
    sprintf(loss_log, "%s/%s.loss.log", param_.log_path().c_str(), prototxt + i + 1);
    sprintf(loss_jpg, "%s/%s.loss.jpg", param_.log_path().c_str(), prototxt + i + 1);
    sprintf(debug_jpg, "%s/%s.debug.jpg", param_.log_path().c_str(), prototxt + i + 1);
    create_dir_recursive(param_.log_path().c_str(), true);
  }
  else if (strlen(param_.net().c_str()) != 0) { 
    sprintf(loss_log, "%s.loss.log", param_.net().c_str()); 
    sprintf(loss_jpg, "%s.loss.jpg", param_.net().c_str()); 
    sprintf(debug_jpg, "%s.debug.jpg", param_.net().c_str());
  }
  else if (strlen(param_.train_net().c_str()) != 0) { 
    sprintf(loss_log, "%s.loss.log", param_.train_net().c_str()); 
    sprintf(loss_jpg, "%s.loss.jpg", param_.train_net().c_str()); 
    sprintf(debug_jpg, "%s.debug.jpg", param_.net().c_str());
  }
  else { 
    sprintf(loss_log, "caffe.loss.log");
    sprintf(loss_jpg, "caffe.loss.jpg"); 
    sprintf(debug_jpg, "caffe.debug.jpg", param_.net().c_str());
  }
  char backup_file[256];
  time_t curtime = time(NULL);
  if (boost::filesystem::exists(boost::filesystem::path(loss_log))) {
	  sprintf(backup_file, "%s.backup_%ld.log", loss_log, curtime);
	  boost::filesystem::copy_file(boost::filesystem::path(loss_log), boost::filesystem::path(backup_file));
  }
  if (boost::filesystem::exists(boost::filesystem::path(loss_jpg))) {
	  sprintf(backup_file, "%s.backup_%ld.jpg", loss_jpg, curtime);
	  boost::filesystem::copy_file(boost::filesystem::path(loss_jpg), boost::filesystem::path(backup_file));
  }
  if (boost::filesystem::exists(boost::filesystem::path(debug_jpg))) {
	  sprintf(backup_file, "%s.backup_%ld.jpg", debug_jpg, curtime);
	  boost::filesystem::rename(boost::filesystem::path(debug_jpg), boost::filesystem::path(backup_file));
  }
  init_log(loss_log);
  canv_loss.create(canv_loss_h, canv_loss_w, CV_8UC3);
  canv_loss = cv::Scalar(0xFF, 0xFF, 0xFF);
  canv_debug.create(canv_debug_h, canv_debug_w, CV_8UC3);
  canv_debug = cv::Scalar(0xFF, 0xFF, 0xFF);
  dsp_thread_run = 1;
  boost::thread dsp_thread = boost::thread(boost::bind(&Solver<Dtype>::dsp_loss_thread, this));
  for (int i = 0; i < stack_cnt * 2; i++) loss_mask[i] = 1;
  memset(plot_mask, 0, sizeof(plot_mask));
  memset(train_label, 0, sizeof(train_label));
  memset(test_label, 0, sizeof(test_label));
  plot_mask[0] = 1;
  iter_tick0 = 0;
  iter_tick1 = 0;
  cur_lr = 0;
  refresh = 0;
  fp_log = NULL;
#endif
}

#ifdef DSP_LOSS
static void get_time_string(int sec, char* str_time) {
	int min = sec / 60 % 60;
	int hour = sec / 60 / 60 % 24;
	int day = sec / 60 / 60 / 24;
	if (day > 0) {
		sprintf(str_time, "%s %d days", str_time, day);
		sprintf(str_time, "%s, %d hours", str_time, hour);
	}
	else if (hour > 0) {
		sprintf(str_time, "%s %d hours", str_time, hour);
		sprintf(str_time, "%s, %d minutes", str_time, min);
	}
	else if (min > 0) {
		sprintf(str_time, "%s %d minutes", str_time, min);
		sprintf(str_time, "%s, %d seconds", str_time, sec % 60);
	}
	else {
		sprintf(str_time, "%s %d seconds", str_time, sec % 60);
	}
}

template <typename Dtype>
void Solver<Dtype>::dsp_loss_thread() {
  while (dsp_thread_run) {
    if (refresh) {
	  draw_all();
	  cv::imshow(loss_jpg, canv_loss);
	  //cv::imshow(debug_jpg, canv_debug);
	  refresh = 0;
	}
	
    int key = cv::waitKey(30);
	if (key >= '1' && key <= '9') {
	  loss_mask[key - '1'] = !loss_mask[key - '1'];
	  refresh = 1;
	}
	else if (key == 'g' || key == 'G') {
	  plot_mask[0] = !plot_mask[0];
	  refresh = 1;
	}
  }
}

template <typename Dtype>
void Solver<Dtype>::init_log(const char* loss_log) {
  FILE* fp = fopen(loss_log, "r");
  if (fp) {
    trloss_stack.clear();
    teloss_stack.clear();
    char line[256], *buf, type[32];
	fgets(line, 256, fp);
	if (strncmp(line, DSP_LOSS_VER, strlen(DSP_LOSS_VER)) == 0) {
      while (fgets(line, 256, fp)) {
        LSTACK stack;
        buf = strtok(line, ", ");
        sscanf(buf, "%d", &stack.iter);
        buf = strtok(NULL, ", ");
        sscanf(buf, "%s", &type);
	    buf = strtok(NULL, ", ");
	    sscanf(buf, "%d", &stack.lcnt);
	    for (int k = 0; k < stack.lcnt; k++) {
		  buf = strtok(NULL, ", ");
		  sscanf(buf, "%f", &stack.loss[k]);
	    }
	    if (strcmp(type, "tr") == 0)
		  trloss_stack.push_back(stack);
	    else if (strcmp(type, "te") == 0)
		  teloss_stack.push_back(stack);
      }
	  fclose(fp);
	}
	else {
	  fclose(fp);
	  fp = fopen(loss_log, "w");
	  fprintf(fp, "%s\n", DSP_LOSS_VER);
	  fclose(fp);
	}    
  }
}

template <typename Dtype>
void Solver<Dtype>::open_log() {
  if (fp_log) return;
  if (trloss_stack.size() == 0) {
	fp_log = fopen(loss_log, "w");
	fprintf(fp_log, "%s\n", DSP_LOSS_VER);
  }
  else if (trloss_stack.back().iter < iter_) {
    fp_log = fopen(loss_log, "a");
  }
  else {
    fp_log = fopen(loss_log, "w");
	fprintf(fp_log, "%s\n", DSP_LOSS_VER);
	int trloss_size = trloss_stack.size();
	int teloss_size = teloss_stack.size();
	int i = 0, j = 0;
	for ( ; i < trloss_size; i++) {
	  if (trloss_stack[i].iter < iter_) {	    
		if (j < teloss_size && teloss_stack[j].iter == trloss_stack[i].iter) {
		  fprintf(fp_log, "%d, te, %d", teloss_stack[j].iter, teloss_stack[j].lcnt);
		  for (int k = 0; k < teloss_stack[j].lcnt; k++) fprintf(fp_log, ", %f", teloss_stack[j].loss[k]);
		  fprintf(fp_log, "\n");
		  j++;
		}
		fprintf(fp_log, "%d, tr, %d", trloss_stack[i].iter, trloss_stack[i].lcnt);
		for (int k = 0; k < trloss_stack[i].lcnt; k++) fprintf(fp_log, ", %f", trloss_stack[i].loss[k]);
		fprintf(fp_log, "\n");
	  }
	  else {
		break;		
	  }
	}
	for ( ; i < trloss_size; i++) {
	  trloss_stack.pop_back();
	}
	for ( ; j < teloss_size; j++) {
	  teloss_stack.pop_back();
	}
  }
}

template <typename Dtype>
void Solver<Dtype>::close_log() {
  if (fp_log) fclose(fp_log);
}

template <typename Dtype>
void Solver<Dtype>::init_test_net() {
	CHECK(Caffe::root_solver());
	string source;
	NetParameter net_param;
	if (param_.has_net()) {
		source = "net file: " + param_.net();
		ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
		test_nets_.resize(1);
		NetState net_state;
		net_state.set_phase(TEST);
		net_state.MergeFrom(net_param.state());
		net_param.mutable_state()->CopyFrom(net_state);
		LOG(INFO)
			<< "Creating test net (#" << 0 << ") specified by " << source;
		if (Caffe::root_solver()) {
			test_nets_[0].reset(new Net<Dtype>(net_param));
		}
		else {
			test_nets_[0].reset(new Net<Dtype>(net_param,
				root_solver_->test_nets_[0].get()));
		}
		test_nets_[0]->set_debug_info(param_.debug_info());
	}
}

template <typename Dtype>
void Solver<Dtype>::write_telog2(float v1, float v2) {
  open_log();
  LSTACK stack;
  stack.iter = iter_;
  stack.lcnt = 2;
  stack.loss[0] = v1;
  stack.loss[1] = v2;
  fprintf(fp_log, "%d, te, %d, %f, %f\n", stack.iter, stack.lcnt, v1, v2);  
  teloss_stack.push_back(stack);
  fflush(fp_log);
}

template <typename Dtype>
void Solver<Dtype>::set_train_label(const char* name) {
  for (int i = 0; i < stack_cnt; i++) {
    if (train_label[i][0] == NULL) {
      strcpy(train_label[i], name);
	  break;
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::set_test_label(const char* name) {
  for (int i = 0; i < stack_cnt; i++) {
    if (test_label[i][0] == NULL) {
      strcpy(test_label[i], name);
	  break;
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::copy_log_to(const char* dest) {
  char path[512];
  sprintf(path, "%s.jpg", dest);
  if (boost::filesystem::exists(boost::filesystem::path(path))) boost::filesystem::remove(boost::filesystem::path(path));
  boost::filesystem::copy_file(boost::filesystem::path(loss_jpg), boost::filesystem::path(path));
  sprintf(path, "%s.log", dest);
  if (boost::filesystem::exists(boost::filesystem::path(path))) boost::filesystem::remove(boost::filesystem::path(path));
  boost::filesystem::copy_file(boost::filesystem::path(loss_log), boost::filesystem::path(path));
}

template <typename Dtype>
void Solver<Dtype>::copy_log_from(const char* src) {
  char path[512];
  sprintf(path, "%s.jpg", src);
  if (boost::filesystem::exists(boost::filesystem::path(path))) {
    if (boost::filesystem::exists(boost::filesystem::path(loss_jpg))) boost::filesystem::remove(boost::filesystem::path(loss_jpg));
    boost::filesystem::copy_file(boost::filesystem::path(path), boost::filesystem::path(loss_jpg));
  }
  sprintf(path, "%s.log", src);
  if (boost::filesystem::exists(boost::filesystem::path(path))) {
    if (boost::filesystem::exists(boost::filesystem::path(loss_log))) boost::filesystem::remove(boost::filesystem::path(loss_log));
    boost::filesystem::copy_file(boost::filesystem::path(path), boost::filesystem::path(loss_log));	
  }
  init_log(loss_log);
}

template <typename Dtype>
void Solver<Dtype>::draw_all() {
	if (trloss_stack.size() >= 2) {
		char txt[512];
		float step = (float)canv_loss_plot_w / (trloss_stack.size() - 1);
		float min_step = 4;

		cv::Scalar plot_color[stack_cnt * 2];
		vector<pair<float, Dtype> > plot_xy[stack_cnt * 2];
		Dtype loss_max = 0;
		Dtype loss_min = 1000;
		Dtype ac_max = 0;
		Dtype ac_min = 1000;
		int n_trgraph = trloss_stack[0].lcnt > 2 ? trloss_stack[0].lcnt : 1;
		int n_tegraph = teloss_stack.size() >= 2 ? teloss_stack[0].lcnt : 0;
		for (int k = 0; k < n_trgraph; k++) {
			if (step >= min_step) {
				for (int i = 0; i < trloss_stack.size(); i++) {
					pair<float, Dtype> xy;
					xy.first = step * i;
					xy.second = trloss_stack[i].loss[k];
					plot_xy[k].push_back(xy);
					if (loss_mask[k]) {
						if (xy.second > loss_max) loss_max = xy.second;
						if (xy.second < loss_min) loss_min = xy.second;
					}
				}
			}
			else {
				int plot_cnt = canv_loss_plot_w / min_step + 1;
				float step_size = (float)trloss_stack.size() / plot_cnt;
				for (int i = 0; i < plot_cnt; i++) {
					pair<float, Dtype> xy;
					xy.first = min_step * i;
					xy.second = 0;
					int startj = (int)(i * step_size + 0.5);
					int endj = i < plot_cnt - 1 ? (int)((i + 1) * step_size + 0.5) : trloss_stack.size();
					for (int j = startj; j < endj; j++) xy.second += trloss_stack[j].loss[k];
					xy.second /= endj - startj;
					plot_xy[k].push_back(xy);
					if (loss_mask[k]) {
						if (xy.second > loss_max) loss_max = xy.second;
						if (xy.second < loss_min) loss_min = xy.second;
					}
				}
			}
			plot_color[k] = cv::Scalar((k % 3) == 2 ? 255 / (k / 3 + 1) : 0, (k % 3) == 1 ? 255 / (k / 3 + 1) : 0, (k % 3) == 0 ? 255 / (k / 3 + 1) : 0);
		}		

		for (int k = 0; k < n_tegraph; k++) {
			if (loss_mask[n_trgraph + k]) {
				Dtype _max = 0;
				Dtype _min = 1000;
				for (int i = 0; i < teloss_stack.size(); i++) {
					if (teloss_stack[i].loss[k] > _max) _max = teloss_stack[i].loss[k];
					if (teloss_stack[i].loss[k] < _min) _min = teloss_stack[i].loss[k];
				}
				if (_max <= 1) {
					if (_max > ac_max) ac_max = _max;
					if (_min < ac_min) ac_min = _min;
					loss_mask[n_trgraph + k] = 2;
				}
				else {
					if (_max > loss_max) loss_max = _max;
					if (_min < loss_min) loss_min = _min;
					loss_mask[n_trgraph + k] = 1;
				}
			}
			plot_color[n_trgraph + k] = cv::Scalar(((n_trgraph + k) % 3) == 2 ? 255 / ((n_trgraph + k) / 3 + 1) : 0, ((n_trgraph + k) % 3) == 1 ? 255 / ((n_trgraph + k) / 3 + 1) : 0, ((n_trgraph + k) % 3) == 0 ? 255 / ((n_trgraph + k) / 3 + 1) : 0);
		}
		if (loss_max < loss_min + 0.01) loss_max = loss_min + 0.01;
		if (ac_max < ac_min + 0.01) ac_max = ac_min + 0.01;

		canv_loss = cv::Scalar(0xFF, 0xFF, 0xFF);
		cv::line(canv_loss, cv::Point(canv_loss_plot_l, canv_loss_plot_t), cv::Point(canv_loss_plot_l, canv_loss_plot_b + 6), cv::Scalar(0, 0, 0), 1);
		cv::line(canv_loss, cv::Point(canv_loss_plot_l - 6, canv_loss_plot_b), cv::Point(canv_loss_plot_r + 6, canv_loss_plot_b), cv::Scalar(0, 0, 0), 1);
		cv::line(canv_loss, cv::Point(canv_loss_plot_r, canv_loss_plot_t), cv::Point(canv_loss_plot_r, canv_loss_plot_b + 6), cv::Scalar(0, 0, 0), 1);

		int plot_row = 10;
		int plot_row_step = canv_loss_plot_h / plot_row;
		for (int i = 0; i < plot_row; i++) {
			cv::line(canv_loss, cv::Point(canv_loss_plot_l - 6, canv_loss_plot_t + i * plot_row_step), cv::Point(canv_loss_plot_l, canv_loss_plot_t + i * plot_row_step), cv::Scalar(0, 0, 0), 1);
			cv::line(canv_loss, cv::Point(canv_loss_plot_l - 3, canv_loss_plot_t + 20 + i * plot_row_step), cv::Point(canv_loss_plot_l, canv_loss_plot_t + 20 + i * plot_row_step), cv::Scalar(0, 0, 0), 1);
			cv::line(canv_loss, cv::Point(canv_loss_plot_r, canv_loss_plot_t + i * plot_row_step), cv::Point(canv_loss_plot_r + 6, canv_loss_plot_t + i * plot_row_step), cv::Scalar(0, 0, 0), 1);
			cv::line(canv_loss, cv::Point(canv_loss_plot_r, canv_loss_plot_t + 20 + i * plot_row_step), cv::Point(canv_loss_plot_r + 3, canv_loss_plot_t + 20 + i * plot_row_step), cv::Scalar(0, 0, 0), 1);
			if (plot_mask[0]) {
				for (int j = 0; j < canv_loss_plot_w / 2; j += 2) {
					cv::line(canv_loss, cv::Point(canv_loss_plot_l + 2 * j, canv_loss_plot_t + i * plot_row_step), cv::Point(canv_loss_plot_l + 2 * j + 2, canv_loss_plot_t + i * plot_row_step), cv::Scalar(0, 0, 0), 1, 8);
					cv::line(canv_loss, cv::Point(canv_loss_plot_l + 2 * j, canv_loss_plot_t + 20 + i * plot_row_step), cv::Point(canv_loss_plot_l + 2 * j + 2, canv_loss_plot_t + 20 + i * plot_row_step), cv::Scalar(0, 0, 0), 1, 8);
				}
			}
			sprintf(txt, "%.2f", loss_min + (loss_max - loss_min)* (1 - (float)i / plot_row));
			cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_l - 45, canv_loss_plot_t + i * plot_row_step + 5), 2, 0.5, cv::Scalar(0, 0, 0));
			if (n_tegraph > 0) {
				sprintf(txt, "%.2f", (ac_min + (ac_max - ac_min) * (1 - (float)i / plot_row)) * 100);
				cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_r + 10, canv_loss_plot_t + i * plot_row_step + 5), 2, 0.5, cv::Scalar(0, 0, 0));
			}
		}
		sprintf(txt, "%.2f", loss_min);
		cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_l - 45, canv_loss_plot_b + 5), 2, 0.5, cv::Scalar(0, 0, 0));
		if (n_tegraph > 0) {
			sprintf(txt, "%.2f", ac_min * 100);
			cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_r + 10, canv_loss_plot_b + 5), 2, 0.5, cv::Scalar(0, 0, 0));
		}

		for (int i = 0; i < plot_xy[0].size() - 1; i++) {
			for (int k = 0; k < n_trgraph; k++) {
				if (loss_mask[k]) {
					cv::Point p1(canv_loss_plot_l + plot_xy[k][i].first, canv_loss_plot_t + canv_loss_plot_h * (1 - (plot_xy[k][i].second - loss_min) / (loss_max - loss_min)));
					cv::Point p2(canv_loss_plot_l + plot_xy[k][i + 1].first, canv_loss_plot_t + canv_loss_plot_h * (1 - (plot_xy[k][i + 1].second - loss_min) / (loss_max - loss_min)));
					cv::line(canv_loss, p1, p2, plot_color[k], 1);
				}
			}
			int draw_iter_label = 0;
			int cur_iter = trloss_stack.front().iter + (trloss_stack.back().iter - trloss_stack.front().iter) * (i + 1) / (plot_xy[0].size() - 1);
			int iter_cnt = trloss_stack.back().iter - trloss_stack.front().iter;
			if (plot_xy[0].size() - 1 < 10) {
				draw_iter_label = 1;
			}
			else if ((i + 1) % (int)((plot_xy[0].size() - 1) / 10.0 + 0.5) == 0) {
				draw_iter_label = 1;
			}
			if (draw_iter_label) {
				if (cur_iter >= 10000) {
					sprintf(txt, "%dk", cur_iter / 1000);
				}
				else if (cur_iter >= 1000) {
					sprintf(txt, "%.1fk", cur_iter / 1000.0);
				}
				else {
					sprintf(txt, "%d", cur_iter);
				}
				cv::line(canv_loss, cv::Point(canv_loss_plot_l + plot_xy[0][i + 1].first, canv_loss_plot_b), cv::Point(canv_loss_plot_l + plot_xy[0][i + 1].first, canv_loss_plot_b + 3), cv::Scalar(0, 0, 0), 1);
				cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_l + plot_xy[0][i + 1].first - 10, canv_loss_plot_b + 20), 2, 0.5, cv::Scalar(0, 0, 0));
			}
		}

		sprintf(txt, "%d", trloss_stack.front().iter);
		cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_l - 10, canv_loss_plot_b + 20), 2, 0.5, cv::Scalar(0, 0, 0));
		sprintf(txt, "%d", trloss_stack.back().iter);
		cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_r - 10, canv_loss_plot_b + 35), 2, 0.5, cv::Scalar(0, 0, 0));

		for (int k = 0; k < n_tegraph; k++) {
			if (loss_mask[n_trgraph + k]) {
				Dtype _max = loss_mask[n_trgraph + k] == 1 ? loss_max : ac_max;
				Dtype _min = loss_mask[n_trgraph + k] == 1 ? loss_min : ac_min;
				int te_w = round((float)canv_loss_plot_w * (teloss_stack.back().iter - teloss_stack.front().iter) / (trloss_stack.back().iter - trloss_stack.front().iter));
				for (int i = 0; i < teloss_stack.size() - 1; i++) {
					cv::Point p1(canv_loss_plot_l + (int)((float)te_w / (teloss_stack.size() - 1) * i + 0.5), canv_loss_plot_t + canv_loss_plot_h * (1 - (teloss_stack[i].loss[k] - _min) / (_max - _min)));
					cv::Point p2(canv_loss_plot_l + (int)((float)te_w / (teloss_stack.size() - 1) * (i + 1) + 0.5), canv_loss_plot_t + canv_loss_plot_h * (1 - (teloss_stack[i + 1].loss[k] - _min) / (_max - _min)));
					cv::line(canv_loss, p1, p2, plot_color[n_trgraph + k], 1);
				}
			}
		}

		sprintf(txt, "Train >>>>>");
		cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_r + 90, canv_loss_plot_t + 15), 2, 0.4, cv::Scalar(0, 0, 0));
		for (int k = 0; k < n_trgraph; k++) {
			cv::line(canv_loss, cv::Point(canv_loss_plot_r + 90, canv_loss_plot_t + 15 * (k + 1) + 12), cv::Point(canv_loss_plot_r + 90 + 50, canv_loss_plot_t + 15 * (k + 1) + 12), plot_color[k], 1);
			if (train_label[k][0] != NULL) {
				sprintf(txt, "%d. %s", k + 1, train_label[k]);
			}
			else if (n_trgraph > 2) {
				if (k == 0) {
					sprintf(txt, "%d. total loss", k + 1);
				}
				else {
					const string& output_name = net_->blob_names()[net_->output_blob_indices()[k - 1]];
					sprintf(txt, "%d. %s", k + 1, output_name.c_str());
				}
			}
			else {
				const string& output_name = net_->blob_names()[net_->output_blob_indices()[k]];
				sprintf(txt, "%d. %s", k + 1, output_name.c_str());
			}
			cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_r + 90 + 50 + 10, canv_loss_plot_t + 15 * (k + 2)), 2, 0.4, cv::Scalar(0, 0, 0));
		}

		sprintf(txt, "Test  >>>>>");
		cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_r + 90, canv_loss_plot_t + 15 * (n_trgraph + 2)), 2, 0.4, cv::Scalar(0, 0, 0));
		for (int k = 0; k < n_tegraph; k++) {
			cv::line(canv_loss, cv::Point(canv_loss_plot_r + 90, canv_loss_plot_t + 15 * (n_trgraph + k + 2) + 12), cv::Point(canv_loss_plot_r + 90 + 50, canv_loss_plot_t + 15 * (n_trgraph + k + 2) + 12), plot_color[n_trgraph + k], 1);
			if (test_label[k][0] != NULL) {
				sprintf(txt, "%d. %s", n_trgraph + k + 1, test_label[k]);
			}
			else if (test_nets_.size() > 0) {
				const shared_ptr<Net<Dtype> >& test_net = test_nets_[0];
				const string& output_name = test_net->blob_names()[test_net->output_blob_indices()[k]];
				sprintf(txt, "%d. %s", n_trgraph + k + 1, output_name.c_str());
			}
			else {
				sprintf(txt, "%d. test #%d", n_trgraph + k + 1, k + 1);
			}
			cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_r + 90 + 50 + 10, canv_loss_plot_t + 15 * (n_trgraph + k + 3)), 2, 0.4, cv::Scalar(0, 0, 0));
		}

		/*
		for (int i = 0; i < net_->layers().size(); i++) {
		Layer<Dtype>& layer = *net_->layers()[i];
		if (strcmp(layer.type(), "Convolution") == 0) {
		Blob<Dtype>& weight = *layer.blobs()[0];
		Blob<Dtype>& bias = *layer.blobs()[1];
		int out_ch = weight.shape()[0];
		int in_ch = weight.shape()[1];
		int kernel_h = weight.shape()[2];
		int kernel_w = weight.shape()[3];
		int dim = out_ch * in_ch * kernel_h * kernel_w;
		if (in_ch != 3) continue;
		const Dtype* weight_data = weight.cpu_data();
		const Dtype* bias_data = bias.cpu_data();
		Dtype max = weight_data[0], min = weight_data[0];
		for (int j = 1; j < dim; j++) {
		if (max < weight_data[j]) max = weight_data[j];
		if (min > weight_data[j]) min = weight_data[j];
		}
		int x, y;
		for (int oc = 0; oc < out_ch; oc++) {
		x = (oc % 8) * kernel_w;
		y = (oc / 8) * kernel_h;
		for (int ky = 0; ky < kernel_h; ky++) {
		for (int kx = 0; kx < kernel_h; kx++) {
		canv_debug.data[((y + ky) * canv_debug.cols + (x + kx)) * 3 + 0] = (int)((weight_data[((oc * in_ch + 0) * kernel_h + ky) * kernel_w + kx] - min) / max * 255 + 0.5);
		canv_debug.data[((y + ky) * canv_debug.cols + (x + kx)) * 3 + 1] = (int)((weight_data[((oc * in_ch + 1) * kernel_h + ky) * kernel_w + kx] - min) / max * 255 + 0.5);
		canv_debug.data[((y + ky) * canv_debug.cols + (x + kx)) * 3 + 2] = (int)((weight_data[((oc * in_ch + 2) * kernel_h + ky) * kernel_w + kx] - min) / max * 255 + 0.5);
		}
		}
		}
		}
		}
		*/

		/*
		for (int i = 0; i < net_->layers().size(); i++) {
			Layer<Dtype>& layer = *net_->layers()[i];
			if (strcmp(layer.type(), "Convolution") == 0 || strcmp(layer.type(), "Pooling") == 0) {
				const vector<Blob<Dtype>*>& bottom = net_->bottom_vecs()[i];
				const vector<Blob<Dtype>*>& top = net_->top_vecs()[i];
				Dtype* map = (Dtype*)calloc(bottom.size[1] * bottom.size[2] * bottom.size[3], sizeof(Dtype));
							

				int id, iy, ix, od, oy, ox, fy, fx;
				int IY, IX, FY, FX;
				for (oy = 0; oy < out_h; oy++) {
					FY = 0;
					IY = oy * s - p;
					if (IY < 0) {
						FY = -IY;
						IY = 0;
					}
					for (ox = 0; ox < out_w; ox++) {
						FX = 0;
						IX = ox * s - p;
						if (IX < 0) {
							FX = -IX;
							IX = 0;
						}
						for (od = 0; od < out_d; od++) {
							uint32_t dst_idx = out_w * (out_h * od + oy) + ox;
							for (id = 0; id < in_d; id++) {
								for (fy = FY, iy = IY; fy < f_h && iy < in_h; fy++, iy++) {
									uint32_t src_idx = in_w * (in_h * id + iy);
									uint32_t filter_idx = f_w * (f_h * (in_d * od + id) + fy);
									for (fx = FX, ix = IX; fx < f_w && ix < in_w; fx++, ix++) {
										in[src_idx + ix] += out[dst_idx];
									}
								}
							}
						}
					}
				}
			}
		}
		*/
		float sec_pre_iter = (float)iter_tick1 / param_.display() / CLOCKS_PER_SEC;
		sprintf(txt, "Speed: %.3f sec/iter", sec_pre_iter);
		cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_l, 20 + 0 * 15), 2, 0.4, cv::Scalar(0, 0, 0));
		sprintf(txt, "Time taken:");
		get_time_string(sec_pre_iter * iter_, txt);
		cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_l, 20 + 1 * 15), 2, 0.4, cv::Scalar(0, 0, 0));
		sprintf(txt, "Time remaining:");
		get_time_string(sec_pre_iter * (param_.max_iter() - iter_), txt);
		cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_l, 20 + 2 * 15), 2, 0.4, cv::Scalar(0, 0, 0));
		sprintf(txt, "LR Policy: %s", param_.lr_policy().c_str());		
		cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_r - 150, 20 + 0 * 15), 2, 0.4, cv::Scalar(0, 0, 0));
		sprintf(txt, "Base LR: %f", param_.base_lr());
		cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_r - 150, 20 + 1 * 15), 2, 0.4, cv::Scalar(0, 0, 0));
		sprintf(txt, "Current LR: %f", cur_lr);
		cv::putText(canv_loss, txt, cv::Point(canv_loss_plot_r - 150, 20 + 2 * 15), 2, 0.4, cv::Scalar(0, 0, 0));
		
		cv::imwrite(loss_jpg, canv_loss);
		//cv::imwrite(debug_jpg, canv_debug);
	}
}
#endif

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  if (Caffe::root_solver()) {
    net_.reset(new Net<Dtype>(net_param));
  } else {
    net_.reset(new Net<Dtype>(net_param, root_solver_->net_.get()));
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  CHECK(Caffe::root_solver());
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    if (Caffe::root_solver()) {
      test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    } else {
      test_nets_[i].reset(new Net<Dtype>(net_params[i],
          root_solver_->test_nets_[i].get()));
    }
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  losses_.clear();
  smoothed_loss_ = 0;

  while (iter_ < stop_iter) {
    // zero-init the params
    net_->ClearParamDiffs();
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())
        && Caffe::root_solver()) {
      TestAll();
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
    }

    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    // accumulate the loss and gradient
    Dtype loss = 0;
    for (int i = 0; i < param_.iter_size(); ++i) {
      loss += net_->ForwardBackward();
    }
    loss /= param_.iter_size();
#if (defined(FORWARD_CHECK) || defined(BACKWARD_CHECK))
    if (!(loss < 80)) {
      LOG_IF(INFO, Caffe::root_solver()) << "Invalid loss value!!";
	  ResetSolver();
      net_->CopyTrainedLayersFrom(net_param_backup);
      continue;
    }
    else if (iter_ % 1000 == 0) {
      net_->ToProto(&net_param_backup, param_.snapshot_diff());
    }
#endif
    // average the loss across iterations for smoothed reporting
    UpdateSmoothedLoss(loss, start_iter, average_loss);
    if (display) {
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << ", loss = " << smoothed_loss_;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
#ifdef DSP_LOSS
	  open_log();
      LSTACK stack;
      stack.iter = iter_;
      stack.lcnt = result.size() + 1;
      stack.loss[0] = smoothed_loss_;	  
	  fprintf(fp_log, "%d, tr, %d, %f", stack.iter, stack.lcnt, stack.loss[0]);
#endif
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
#ifdef DSP_LOSS
          stack.loss[j + 1] = result_vec[k];
		  fprintf(fp_log, ", %f", stack.loss[j + 1]);
#endif
        }
      }
#ifdef DSP_LOSS
      trloss_stack.push_back(stack);
	  fprintf(fp_log, "\n");
	  fflush(fp_log);
	  if (iter_tick0 == 0) {
		iter_tick0 = clock();
	  }
	  else {
		clock_t cur_tick = clock();
		iter_tick1 = cur_tick - iter_tick0;
		iter_tick0 = cur_tick;
	  }
	  refresh = 1;
#endif
    }
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }
    ApplyUpdate();

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;

    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      Snapshot();
#ifdef DSP_LOSS
	  char dest[512];
	  strcpy(dest, Solver<Dtype>::SnapshotFilename(".jpg").c_str());
	  if (boost::filesystem::exists(boost::filesystem::path(dest))) boost::filesystem::remove(boost::filesystem::path(dest));
	  boost::filesystem::copy_file(boost::filesystem::path(loss_jpg), boost::filesystem::path(dest));
	  strcpy(dest, Solver<Dtype>::SnapshotFilename(".log").c_str());
	  if (boost::filesystem::exists(boost::filesystem::path(dest))) boost::filesystem::remove(boost::filesystem::path(dest));
	  boost::filesystem::copy_file(boost::filesystem::path(loss_log), boost::filesystem::path(dest));
#endif
    }
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  int start_iter = iter_;
  Step(param_.max_iter() - iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_.display() && iter_ % param_.display() == 0) {
    int average_loss = this->param_.average_loss();
    Dtype loss;
    net_->Forward(&loss);

    UpdateSmoothedLoss(loss, start_iter, average_loss);

    LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    Test(test_net_id);
  }
}

template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO)     << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
#ifdef DSP_LOSS
  open_log();
  LSTACK stack;
  stack.iter = iter_;
  stack.lcnt = test_score.size();
  fprintf(fp_log, "%d, te, %d", stack.iter, stack.lcnt);
#endif
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
#ifdef DSP_LOSS
	stack.loss[i] = mean_score;
	fprintf(fp_log, ", %f", stack.loss[i]);
#endif
  }
#ifdef DSP_LOSS
  teloss_stack.push_back(stack);
  fprintf(fp_log, "\n");
  fflush(fp_log);
#endif
}

template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {
  if (Caffe::root_solver() && param_.snapshot()) {
    CHECK(param_.has_snapshot_prefix())
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
          << param_.snapshot_prefix() << "'.  Make sure "
          << "that the directory exists and is writeable.";
    }
  }
}

template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string extension) {
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  CHECK(Caffe::root_solver());
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
    int average_loss) {
  if (losses_.size() < average_loss) {
    losses_.push_back(loss);
    int size = losses_.size();
    smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
  } else {
    int idx = (iter_ - start_iter) % average_loss;
    smoothed_loss_ += (loss - losses_[idx]) / average_loss;
    losses_[idx] = loss;
  }
}

INSTANTIATE_CLASS(Solver);

}  // namespace caffe
