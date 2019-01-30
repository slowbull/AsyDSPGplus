
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
#ifndef _DEFINES_
#define _DEFINES_

#include <math.h>
#include <omp.h>
#include <cstdlib>
#include <thread>
#include <map>
#include <set>
#include <cstring>
#include <cstdlib>
#include <gflags/gflags.h>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <time.h>
#include <sys/time.h>
#include "Datapoint/Datapoint.h"
#include "Gradient/Gradient.h"
#include "DatasetReader.h"
#include "DatapointPartitions/DatapointPartitions.h"
#include "Partitioner/Partitioner.h"
#include "Partitioner/BasicPartitioner.h"
#include "Model/Model.h"

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#include <sys/time.h>
int clock_gettime(int /*clk_id*/, struct timespec* t) {
    struct timeval now;
    int rv = gettimeofday(&now, NULL);  // return 0 for success.
    if (rv) return rv;
    t->tv_sec  = now.tv_sec;
    t->tv_nsec = now.tv_usec * 1000;
    return 0;
}
#define CLOCK_MONOTONIC 0
#endif


// Timer use std::chrono maybe a faster way.
class Timer {
public:
    struct timespec _start;
    struct timespec _end;
	double elapsed;

    Timer(){
        clock_gettime(CLOCK_MONOTONIC, &_start);
    }
    virtual ~Timer(){}
    inline void Restart(){
        clock_gettime(CLOCK_MONOTONIC, &_start);
    }
    inline float Elapsed(){
        clock_gettime(CLOCK_MONOTONIC, &_end);
        return (_end.tv_sec - _start.tv_sec) + (_end.tv_nsec - _start.tv_nsec) / 1000000000.0;
    }

	inline void Tick(){
        clock_gettime(CLOCK_MONOTONIC, &_start);
	}

	inline void Tock(){
        clock_gettime(CLOCK_MONOTONIC, &_end);
        elapsed = (_end.tv_sec - _start.tv_sec) + (_end.tv_nsec - _start.tv_nsec) / 1000000000.0;
	}
};

void pin_to_core(size_t core) {
#ifdef _GNU_SOURCE
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset); // clear set, so that it contains no cpu.
    CPU_SET(core, &cpuset); // add cpu  to cpuset
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

DEFINE_string(data_file, "blank", "Input data file.");
DEFINE_string(model_block, "", "model block file.");
DEFINE_string(model_snapshot, "", "model snapshot to be loaded.");
DEFINE_int32(n_epochs, 100, "Number of passes of data in training.");
DEFINE_int32(n_threads, 2, "Number of threads in parallel during training.");
DEFINE_int32(base_thread, 0, "base thread.");
DEFINE_int32(mini_batch, 1, "Mini batch size during training.");
DEFINE_double(learning_rate, .001, "Learning rate.");
DEFINE_double(learning_rate_dec, 0, "Learning rate decreasing in each epoch. 1/(1+epoch)^learning_rate_dec.");
DEFINE_double(q_memory, 0, "for q-memory svrg. rate of data before full gradient computation. if q_memory = 1, it is equal to SVRG"); // generval svrg
DEFINE_double(l1_lambda, 0, "regularization parameter for l1 norm.");
DEFINE_double(l2_lambda, 0, "regularization parameter for l2 norm.");
DEFINE_double(l0_k, 0, "l0 norm regularization. k is the number of non-zero elements");
DEFINE_bool(print_loss_per_epoch, false, "Should compute and print loss every epoch.");
DEFINE_bool(sparse_update, false, "in each iteration, only update the nonzero coordinates in x.");


DEFINE_bool(shuffle_datapoints, true, "Shuffle datapoints before training.");

// Flags for training types.
DEFINE_bool(hogwild_trainer, false, "Hogwild training method (parallel).");
DEFINE_bool(batch_trainer, false, "batch training method (parallel).");

// Flags for updating types.
DEFINE_bool(dense_svrg, false, "Use the dense SVRG.");
DEFINE_bool(dense_saga, false, "Use the dense Saga.");
DEFINE_bool(dense_sgd, false, "Use the dense SGD.");
DEFINE_bool(batch_grad, false, "Use the batch gradient method.");

// MISC flags.
DEFINE_int32(random_range, 100, "Range of random numbers for initializing the model.");

#include "Updater/Updater.h"
#include "Updater/DenseSVRGUpdater.h"
#include "Updater/DenseSGDUpdater.h"
#include "Updater/GDUpdater.h"

#include "Trainer/Trainer.h"
#include "Trainer/BatchTrainer.h"
#include "Trainer/HogwildTrainer.h"

#include "Datapoint/LIBSVMDatapoint.h"

#include "Model/LOGISTICL2L1Model.h"

#endif
