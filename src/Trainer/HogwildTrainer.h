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
#ifndef _HOGWILD_TRAINER_
#define _HOGWILD_TRAINER_

class HogwildTrainer : public Trainer {
public:
    HogwildTrainer() {}
    ~HogwildTrainer() {}

    TrainStatistics Train(Model *model, const std::vector<Datapoint *> & datapoints, Updater *updater) override {
	// Partition.
	BasicPartitioner partitioner;
	DatapointPartitions partitions = partitioner.Partition(datapoints, FLAGS_n_threads);

	model->SetUpWithPartitions(partitions);
	updater->SetUpWithPartitions(partitions);

	// Default datapoint processing ordering.
	// [thread][batch][index].
	std::vector<std::vector<std::vector<int> > > per_batch_datapoint_order(FLAGS_n_threads);
	for (int thread = 0; thread < FLAGS_n_threads; thread++) {
	    per_batch_datapoint_order[thread].resize(partitions.NumBatches());
	    for (int batch = 0; batch < partitions.NumBatches(); batch++) {
		per_batch_datapoint_order[thread][batch].resize(partitions.NumDatapointsInBatch(thread, batch));
		for (int index = 0; index < partitions.NumDatapointsInBatch(thread, batch); index++) {
		    per_batch_datapoint_order[thread][batch][index] = index;
		}
	    }
	}

	// Keep track of statistics of training.
	TrainStatistics stats;

	std::default_random_engine generator (0);
	std::uniform_real_distribution<double> distribution (FLAGS_q_memory*0.1, FLAGS_q_memory*1.9); // uniform real distribution

	// Train.
	Timer gradient_timer;
	int total_counts = 0, index_counts = 0;

	printf("Iterations:     Epoch: 	Time(s): Loss:   Evaluation(MSE for Regression, AUC for classificaton): \n");
	for (int epoch = 0; epoch < FLAGS_n_epochs; epoch++) {
		srand(epoch);


		total_counts += index_counts * FLAGS_n_threads;
		printf("%d    ", total_counts);

		// compute loss and print working time.
	    this->EpochBegin(epoch, gradient_timer, model, datapoints, &stats);

	    // Random per batch datapoint processing.  ?
	    if (FLAGS_random_per_batch_datapoint_processing) {
		for (int thread = 0; thread < FLAGS_n_threads; thread++) {
		    int batch = 0;
		    for (int index = 0; index < partitions.NumDatapointsInBatch(thread, batch); index++) {
			per_batch_datapoint_order[thread][batch][index] = rand() % partitions.NumDatapointsInBatch(thread, batch);
		    }
		}
	    }

		gradient_timer.Tick();
	    updater->EpochBegin();

		// for q memory
		index_counts = partitions.NumDatapointsInBatch(0, 0); 
		if (FLAGS_q_memory != 0) {
			index_counts = int(index_counts * distribution(generator));
		}

#pragma omp parallel for schedule(static, 1)
	    for (int thread = 0; thread < FLAGS_n_threads; thread++) {
		int batch = 0; // Hogwild only has 1 batch.
		int index;
		double learning_rate;

		for (int index_count = 0; index_count < index_counts / FLAGS_mini_batch; index_count++) {
			int block_index = rand() % (model->model_block.size()-1);
			for(int batch_iter = 0; batch_iter < FLAGS_mini_batch; batch_iter++){
		    	index = per_batch_datapoint_order[thread][batch][rand()%partitions.NumDatapointsInBatch(thread, batch)];
				learning_rate = FLAGS_learning_rate / pow(epoch+1, FLAGS_learning_rate_dec);
		    	updater->Update(model, partitions.GetDatapoint(thread, batch, index), block_index, learning_rate);
			}
			updater->ApplyProximalOperator(block_index, partitions.GetDatapoint(thread, batch, index), FLAGS_l1_lambda * learning_rate);
		}
	    }
	    updater->EpochFinish();
		gradient_timer.Tock();

	}
	model->StoreModel();
	return stats;
    }
};

#endif
