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

#ifndef _LOGISTICL2L1MODEL_
#define _LOGISTICL2L1MODEL_

#include <sstream>
#include <math.h>
#include "Model.h"


class LOGISTICL2L1Model : public Model {
 private:
    int n_coords;
    std::vector<double> model;
	int postive=0;
	int negative=0;

    void Initialize(const std::string &input_line) {
		// Expect a single number with n_coords.
		std::stringstream input(input_line);
		input >> n_coords;

		// Initialize model.
		model.resize(n_coords);
		std::fill(model.begin(), model.end(), 0);

		if(!FLAGS_model_snapshot.empty()){
			std::ifstream data_file_input(FLAGS_model_snapshot);
			if (!data_file_input) {
	    		std::cerr << "Model_snapshot: Could not open model snapshot file ! " << FLAGS_model_snapshot << std::endl;
	    		exit(0);
			}
			std::string model_snapshot;
			double weight;
			std::getline(data_file_input, model_snapshot);
			std::stringstream input(model_snapshot);
			model.resize(0);
			while(input){
				input >> weight;
				if(!input)
					break;
				model.push_back(weight);
			}
			if(model.size() != n_coords){
				std::cerr << "model snapshot size " <<  model.size() << " is not consistent with model size" << n_coords << " !" << std::endl;
				exit(0);
			}
		}
    }
 public:

    LOGISTICL2L1Model(const std::string &input_line) {
		Initialize(input_line);
		Model::SetModelBlock(FLAGS_model_block, n_coords);
    }

	// setup postive and negative numbers.
    void SetUp(const std::vector<Datapoint *> &datapoints) override {
		for(int i=0; i<datapoints.size(); i++){
			if(datapoints[i]->GetLabel()==1){
				postive += 1;
			}
			else{
				negative += 1;
			}
		}
    }

	template<typename T>
		std::vector<int> sort_indexes(const std::vector<T> & v){
			std::vector<int> idx(v.size());
			iota(idx.begin(), idx.end(), 0);
			std::sort(idx.begin(), idx.end(), [&v](int i1, int i2){return v[i1] < v[i2];});
			return idx;
		}

    double ComputeLoss(const std::vector<Datapoint *> &datapoints, double& auc) override {
		double loss = 0;
		std::vector<double> prob(datapoints.size(), 0);
		#pragma omp parallel for schedule(static, 1) reduction(+:loss)
		for (int i = 0; i < datapoints.size(); i++) {
			Datapoint *datapoint = datapoints[i];
			double cross_product = 0;
			for (int j = 0; j < datapoint->GetCoordinates().size(); j++) {
			int index = datapoint->GetCoordinates()[j];
			double weight = datapoint->GetWeights()[j];
			cross_product += model[index] * weight;
			}
			prob[i] = cross_product;
			loss += log(1 + exp(-cross_product * datapoint->GetLabel()));
		}

		// caculating auc!!!
		std::vector<int> idx = sort_indexes(prob);	
		double sum = 0;
		for(int i = 0; i < datapoints.size(); i++){
			double element = (datapoints[idx[i]]->GetLabel() + 1) / 2;
			sum += element;
			prob[i] =  sum;
		}
		auc = 0;
		double y2 = 1, y1 = 0, x2 = 1, x1 = 0;
		for(int i=0; i < datapoints.size(); i++){
			y1 = (postive - prob[i]) / postive;
			x1 = 1 - ( i + 1 - prob[i]) / negative;
			auc += 0.5 * (y2+y1)*(x2-x1);
			x2 = x1;
			y2 = y1;
		}
			
		return loss / datapoints.size() + ComputeRegularization(); 
	}

	double ComputeRegularization(){
		double regloss = 0;
		for(int i=0; i<model.size(); i++){
			regloss += std::abs(model[i]) * FLAGS_l1_lambda + 0.5 * std::pow(model[i], 2) * FLAGS_l2_lambda;	
		}
		return regloss;
	}

	virtual double ProximalOperator(std::vector<double> &model_data, double gamma, int block_index, Datapoint *datapoint){
	int coordinate_size = 1;
	if (FLAGS_sparse_update) {
		for(int i = 0; i < datapoint->GetCoordinates().size(); i++){
			int index = datapoint->GetCoordinates()[i];
			if( index < model_block[block_index] )
				continue;
			else if( index >= model_block[block_index+1] )
				break;
			for (int j = 0; j < coordinate_size; j++) {
				double val = model_data[index * coordinate_size + j];
				double sign = val > 0 ? 1: -1;	
				model_data[index * coordinate_size + j] = sign * fmax( std::abs(val) - gamma, 0);
			}
		}
	}
	else {
		for (int i = model_block[block_index]; i < model_block[block_index+1]; i++) {
			int index = i;
		  	for (int j = 0; j < coordinate_size; j++) {
				double val = model_data[index * coordinate_size + j];
				double sign = val > 0 ? 1: -1;	
				model_data[index * coordinate_size + j] = sign * fmax( std::abs(val) - gamma, 0);
		  	}
		}
	}
	}

	virtual void StoreModel(){
		std::ofstream file("model.out");
		for(int i=0; i<model.size(); i++){
			file << model[i] << " ";
		}
		file.close();
	}

    int NumParameters() override {
	return n_coords;
    }

    int CoordinateSize() override {
	return 1;
    }

    std::vector<double> & ModelData() override {
	return model;
    }

    void PrecomputeCoefficients(Datapoint *datapoint, Gradient *g, std::vector<double> &local_model, int block_index) override {
	if (g->coeffs.size() != n_coords) g->coeffs.resize(n_coords);
	double cp = 0;
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    double weight = datapoint->GetWeights()[i];
	    cp += weight * local_model[index];
	}
	double partial_grad = -datapoint->GetLabel() * exp(-datapoint->GetLabel() * cp) / (1 + exp(-datapoint->GetLabel() * cp)) ;
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
		if(block_index != -1){
			if(index < model_block[block_index])
				continue;
			else if (index >= model_block[block_index+1])
				break;
		}
	    double weight = datapoint->GetWeights()[i];
	    g->coeffs[index] = partial_grad * weight;
	}
    }

    void Lambda(int coordinate, double &out, std::vector<double> &local_model) override {
	out = FLAGS_l2_lambda;
    }

    void Kappa(int coordinate, std::vector<double> &out, std::vector<double> &local_model) override {
	out[0] = 0;
    }

    void H_bar(int coordinate, std::vector<double> &out, Gradient *g, std::vector<double> &local_model) override {
	out[0] = g->coeffs[coordinate];
    }

    ~LOGISTICL2L1Model() {
    }
};

#endif
