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

#ifndef _LIBSVM_DATAPOINT_
#define _LIBSVM_DATAPOINT_

#include <tuple>
#include <sstream>
#include <string.h>
#include <stdlib.h>
#include "Datapoint.h"

class LIBSVMDatapoint : public Datapoint {
 private:
    std::vector<double> weights;
    std::vector<int> coordinates;
	double label;

    void Initialize(const std::string &input_line) {
		// Expected format:
		// label index#1 weight1 index#2 weight2 ...
		char* c_index, *c_weight, *c_label;
		char line[100000];
		strcpy(line, input_line.c_str());
		c_label = strtok(line, " \t");
		label = strtod(c_label, NULL);
		while(1){
			int index;
			double weight;
			c_index = strtok(NULL, ":");
			c_weight = strtok(NULL, " \t");
			if(c_weight==NULL)
				break;
			index = (int) strtol(c_index, NULL, 10);
			coordinates.push_back(index-1);
			weight = strtod(c_weight,NULL);
			weights.push_back(weight);
		}
    }

 public:

    LIBSVMDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) {
	Initialize(input_line);
    }
    ~LIBSVMDatapoint() {} 
    std::vector<double> & GetWeights() override {
	return weights;
    }

    std::vector<int> & GetCoordinates() override {
	return coordinates;
    }

    int GetNumCoordinateTouches() override {
	return coordinates.size();
    }

	double GetLabel(){
	return label;
	}
};

#endif
