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

#ifndef _DENSE_SGD_UPDATER_
#define _DENSE_SGD_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class DenseSGDUpdater : public Updater {
protected:
    REGISTER_THREAD_LOCAL_2D_VECTOR(h_bar);


    void PrepareNu(std::vector<int> &coordinates) override {
	// Nu is 0.
    }

    void PrepareMu(std::vector<int> &coordinates) override {
	// l2_lambda
    }

    void PrepareH(Datapoint *datapoint, Gradient *g, int block_index) override {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<std::vector<double> > &h_bar = GET_THREAD_LOCAL_VECTOR(h_bar);
	if(!FLAGS_sparse_update) 
		std::fill(h_bar.begin(), h_bar.end(), std::vector<double>(h_bar[0].size(),0));
	model->PrecomputeCoefficients(datapoint, g, cur_model, block_index);
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
		if(index < model->model_block[block_index])
			continue;
		else if(index >= model->model_block[block_index+1])
			break;
	    model->H_bar(index, h_bar[index], g, cur_model);
	}
    }

	double Grad(double val, int coordinate, int index_into_coordinate_vector) {
	return Mu(coordinate) * val + Nu(coordinate, index_into_coordinate_vector) - H(coordinate, index_into_coordinate_vector);
	}

    double H(int coordinate, int index_into_coordinate_vector) {
	return -GET_THREAD_LOCAL_VECTOR(h_bar)[coordinate][index_into_coordinate_vector];
    }

    double Nu(int coordinate, int index_into_coordinate_vector) {
	return 0;
    }

    double Mu(int coordinate) {
	return FLAGS_l2_lambda;
    }

 public:
    DenseSGDUpdater(Model *model, std::vector<Datapoint *> &datapoints) : Updater(model, datapoints) {
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(h_bar, model->NumParameters(), model->CoordinateSize());
    }

    ~DenseSGDUpdater() {
    }
};

#endif
