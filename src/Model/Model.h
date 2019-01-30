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
#ifndef _MODEL_
#define _MODEL_

#include<sstream>
#include "../DatapointPartitions/DatapointPartitions.h"


class Model {
 public:
    Model() {}
    Model(const std::string &input_line) {}
    virtual ~Model() {}
	std::vector<int>  model_block; // model_block = [0, 10, 15, 20]  block i \in [model_block[i], model_block[i+1])

    // Computes loss on the model
    virtual double ComputeLoss(const std::vector<Datapoint *> &datapoints, double &evaluation) = 0;

    // Do some set up with the model and datapoints before running gradient descent.
    virtual void SetUp(const std::vector<Datapoint *> &datapoints) {}

    // Do some set up with the model given partitioning scheme before running the trainer.
    virtual void SetUpWithPartitions(DatapointPartitions &partitions) {}

    // Return the number of parameters of the model.
    virtual int NumParameters() = 0;

    // Return the size (the # of doubles) of a single coordinate.
    virtual int CoordinateSize() = 0;

    // Return data to actual model.
    virtual std::vector<double> & ModelData() = 0;

	// return closed form solution for proximal operator.
	virtual double ProximalOperator(std::vector<double> &model_data, double gamma, int block_index, Datapoint *datapoint) = 0;

    // Return some extra data which may be useful to be modified.
    virtual std::vector<double> & ExtraData() {
	// Default: return ModelData.
	return ModelData();
    }

	virtual void SetModelBlock(const std::string &block_file, int model_size){
		if(block_file.empty()){
			model_block.resize(2);
			model_block[0] = 0;
			model_block[1] = model_size;
		}
		else{
			std::ifstream data_file_input(block_file);

			if (!data_file_input) {
	    		std::cerr << "Model_block: Could not open model block file ! " << block_file << std::endl;
	    		exit(0);
			}
			std::string block_line;
			int index;
			model_block.resize(0);
			while(std::getline(data_file_input, block_line)){
				std::stringstream input(block_line);
				while(input){ 
					input >> index;
					if(!input)
						break;
					model_block.push_back(index);
				}
			}
		}
		std::cout << "Model block: ";
		for(auto const &ele:model_block)
			std::cout << ele << " ";
		std::cout << "\n";
	}	

	virtual void StoreModel() = 0;

    // The following are for updates of the form:
    // [∇f(x)] = λx − κ + h(x)
    // See https://arxiv.org/pdf/1605.09721v1.pdf page 20 for more details.
    virtual void PrecomputeCoefficients(Datapoint *datapoint, Gradient *g, std::vector<double> &local_model, int block_index) = 0;
    virtual void Lambda(int coordinate, double &out, std::vector<double> &local_model) = 0;
    virtual void Kappa(int coordinate, std::vector<double> &out, std::vector<double> &local_model) = 0;
    virtual void H_bar(int coordinate, std::vector<double> &out, Gradient *g, std::vector<double> &local_model) = 0;
};

#endif
