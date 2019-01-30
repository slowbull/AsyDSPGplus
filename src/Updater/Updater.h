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

#ifndef _UPDATER_
#define _UPDATER_

#include "../DatapointPartitions/DatapointPartitions.h"
#include "../Gradient/Gradient.h"

// Some macros to declare extra thread-local / global 1d/2d vectors.
// This avoids the use of std::maps, which are very inefficient.
// Gives around a 2-3x speedup over using maps.
#define REGISTER_THREAD_LOCAL_1D_VECTOR(NAME) std::vector<std::vector<double> > NAME ## _LOCAL_
#define REGISTER_THREAD_LOCAL_2D_VECTOR(NAME) std::vector<std::vector<std::vector<double> > > NAME ## _LOCAL_

#define INITIALIZE_THREAD_LOCAL_1D_VECTOR(NAME, N_COLUMNS) {NAME##_LOCAL_.resize(FLAGS_n_threads); for (int i = 0; i < FLAGS_n_threads; i++) NAME ## _LOCAL_[i].resize(N_COLUMNS, 0);}
#define INITIALIZE_THREAD_LOCAL_2D_VECTOR(NAME, N_ROWS, N_COLUMNS) {NAME##_LOCAL_.resize(FLAGS_n_threads); for (int i = 0; i < FLAGS_n_threads; i++) NAME ## _LOCAL_[i].resize(N_ROWS, std::vector<double>(N_COLUMNS, 0));}

#define GET_THREAD_LOCAL_VECTOR(NAME) NAME ## _LOCAL_[omp_get_thread_num()]

#define REGISTER_GLOBAL_1D_VECTOR(NAME) std::vector<double> NAME ## _GLOBAL_
#define REGISTER_GLOBAL_2D_VECTOR(NAME) std::vector<std::vector<double> > NAME ## _GLOBAL_

#define INITIALIZE_GLOBAL_1D_VECTOR(NAME, N_COLUMNS) {NAME ## _GLOBAL_.resize(N_COLUMNS, 0);}
#define INITIALIZE_GLOBAL_2D_VECTOR(NAME, N_ROWS, N_COLUMNS) {NAME ## _GLOBAL_.resize(N_ROWS, std::vector<double>(N_COLUMNS, 0));}

#define GET_GLOBAL_VECTOR(NAME) NAME ## _GLOBAL_

#define REGISTER_THREAD_LOCAL_DOUBLE(NAME) std::vector<double > NAME ## _LOCAL_
#define INITIALIZE_THREAD_LOCAL_DOUBLE(NAME) {NAME##_LOCAL_.resize(FLAGS_n_threads); std::fill(NAME##_LOCAL_.begin(), NAME##_LOCAL_.end(), 0);}

class Updater {
protected:
    // Keep a reference of the model and datapoints, and partition ordering.
    Model *model;
    std::vector<Datapoint *> datapoints;
    DatapointPartitions *datapoint_partitions;

    // Have an array of Gradient objects (stores extra info for Model processing).
    // Have 1 per thread to avoid conflicts.
    Gradient *thread_gradients;

    // A reference to all_coordinates, which indexes all the coordinates of the model.
    std::vector<int> all_coordinates;

    // H, Nu and Mu for updates.
    virtual double H(int coordinate, int index_into_coordinate_vector) = 0;
    virtual double Nu(int coordinate, int index_into_coordinate_vector) = 0;
    virtual double Mu(int coordinate) = 0;

	virtual double Grad(double model_val, int coordinate, int index_into_coordinate_vector) = 0;

    // After calling PrepareNu/Mu/H, for the given coordinates, we expect that
    // calls to Nu/Mu/H are ready.
    virtual void PrepareNu(std::vector<int> &coordinates) = 0;
    virtual void PrepareMu(std::vector<int> &coordinates) = 0;
    virtual void PrepareH(Datapoint *datapoint, Gradient *g, int block_index) = 0;


    virtual void ApplyGradient(Datapoint *datapoint, int block_index, double learning_rate) {
	int coordinate_size = model->CoordinateSize();
	std::vector<double> &model_data = model->ModelData();
	if(FLAGS_sparse_update) {
		for(int i=0; i < datapoint->GetCoordinates().size(); i++){
			int index = datapoint->GetCoordinates()[i];
			if( index < model->model_block[block_index] )
				continue;
			else if( index >= model->model_block[block_index+1] )
				break;
			for (int j = 0; j < coordinate_size; j++) {
				model_data[index * coordinate_size + j] = model_data[index * coordinate_size + j] - Grad(model_data[index * coordinate_size + j], index, j) * learning_rate / FLAGS_mini_batch;
			}
		}
	}
	else {
		for(int i=model->model_block[block_index]; i < model->model_block[block_index+1]; i++){
			int index = i;
			for (int j = 0; j < coordinate_size; j++) {
				model_data[index * coordinate_size + j] = model_data[index * coordinate_size + j] - Grad(model_data[index * coordinate_size + j], index, j) * learning_rate / FLAGS_mini_batch;
			}
		}
    }
	}



public:
    Updater(Model *model, std::vector<Datapoint *> &datapoints) {
	// Create gradients for each thread.
	thread_gradients = new Gradient[FLAGS_n_threads];
	for (int thread = 0; thread < FLAGS_n_threads; thread++) {
	    thread_gradients[thread] = Gradient();
	}
	//separate local variable from class member. should avoid this in coding. 
	this->model = model;

	// Set up 
	this->datapoints = datapoints;

	// Keep an array that has integers 1...n_coords.
	for (int i = 0; i < model->NumParameters(); i++) {
	    all_coordinates.push_back(i);
	}
    }

    Updater() {}
    virtual ~Updater() {
	delete [] thread_gradients;
    }

    // Could be useful to get partitioning info.
    virtual void SetUpWithPartitions(DatapointPartitions &partitions) {
	datapoint_partitions = &partitions;
    }

    // Main update method, which is run by multiple threads.
    virtual void Update(Model *model, Datapoint *datapoint, int block_index, double learning_rate) {
	int thread_num = omp_get_thread_num();
	thread_gradients[thread_num].Clear();
	thread_gradients[thread_num].datapoint = datapoint;

	// After catching up, prepare H and apply the gradient.
	PrepareNu(datapoint->GetCoordinates());
	PrepareMu(datapoint->GetCoordinates());
	PrepareH(datapoint, &thread_gradients[thread_num], block_index);
	ApplyGradient(datapoint, block_index, learning_rate);
    }

	virtual void ApplyProximalOperator(int block_index, Datapoint *datapoint, double gamma){
		std::vector<double> &model_data = model->ModelData();
		model->ProximalOperator(model_data, gamma, block_index, datapoint);
	}

    // Called before epoch begins.
    virtual void EpochBegin() {
    }

    // Called when the epoch ends.
    virtual void EpochFinish() {
    }
};

#endif
