/*
* Copyright 2016 [See AUTHORS file for list of authors]
*
*    Licensed under the Apache License, Version 2.0 (the "License");
*    you may not use this file except in compliance with the License.
*    You may obtain a copy of the License at
*
*        http://www.apache.org/licenses/LICENSE-2.0
*
*    Unless required by applicable law or agreed to in writing, software
*    distributed under the License is distributed on an "AS IS" BASIS,
*    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*    See the License for the specific language governing permissions and
*    limitations under the License.
*/
#ifndef _TRAINER_
#define _TRAINER_

#include <limits.h>
#include <float.h>

DEFINE_bool(random_batch_processing, false, "Process batches in random order. Note this may disrupt catch-up.");
DEFINE_bool(random_per_batch_datapoint_processing, false, "Process datapoints in random order per batch. Note this may disrupt catch-up.");
DEFINE_int32(interval_print, 1, "Interval in which to print the loss.");

// Contains times / losses / etc
struct TrainStatistics {
    std::vector<double> times;
    std::vector<double> losses;
};

typedef struct TrainStatistics TrainStatistics;

class Trainer {
protected:

    void TrackTimeLoss(double cur_time, double cur_loss, TrainStatistics *stats) {
	stats->times.push_back(cur_time);
	stats->losses.push_back(cur_loss);
    }

    void PrintTimeLoss(double cur_time, double cur_loss, int epoch, double cur_eval) {
	printf("%d     %f    %.10f    %f\n", epoch, cur_time, cur_loss, cur_eval);
    }

    void EpochBegin(int epoch, Timer &gradient_timer, Model *model, const std::vector<Datapoint *> &datapoints, TrainStatistics *stats) {
	double cur_time;
    if(stats->times.size()==0) 
		cur_time = 0;
	else
		cur_time = gradient_timer.elapsed + stats->times[stats->times.size()-1];
	double cur_eval = 0; 
	double cur_loss = model->ComputeLoss(datapoints, cur_eval);
	this->TrackTimeLoss(cur_time, cur_loss, stats);
	if (FLAGS_print_loss_per_epoch && epoch % FLAGS_interval_print == 0) {
	    this->PrintTimeLoss(cur_time, cur_loss, epoch, cur_eval);
	}
    }

public:
    Trainer() {
	// Some error checking.
	if (FLAGS_n_threads > std::thread::hardware_concurrency()) {
	    std::cerr << "Trainer: Number of threads is greater than the number of physical cores." << std::endl;
	    //exit(0);
	}

	// Basic set up, like pinning to core, setting number of threads.
	omp_set_num_threads(FLAGS_n_threads);
#pragma omp parallel
	{
	    pin_to_core(omp_get_thread_num()+ FLAGS_base_thread);
	}
    }
    virtual ~Trainer() {}

    // Main training method.
    virtual TrainStatistics Train(Model *model, const std::vector<Datapoint *> & datapoints, Updater *updater) = 0;
};

#endif
