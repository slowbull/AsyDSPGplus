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
#ifndef _Batch_TRAINER_
#define _Batch_TRAINER_

class BatchTrainer : public Trainer {
public:
    BatchTrainer() {}
    ~BatchTrainer() {}

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

	// Train.
	Timer gradient_timer;
	printf("Epoch: 	Time(s): Loss:   Evaluation(MSE for Regression, AUC for classificaton): \n");
	for (int epoch = 0; epoch < FLAGS_n_epochs; epoch++) {
		srand(epoch);

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

		int block_index = rand() % (model->model_block.size()-1);
		double learning_rate = FLAGS_learning_rate / pow(epoch+1, FLAGS_learning_rate_dec);
		updater->Update(model, NULL, block_index, learning_rate);
		updater->ApplyProximalOperator(block_index, NULL, FLAGS_l1_lambda * learning_rate);

	    updater->EpochFinish();
		gradient_timer.Tock();

	}
	model->StoreModel();
	return stats;
    }
};

#endif
