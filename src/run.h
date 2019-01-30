#include <iostream>
#include "defines.h"

template<class MODEL_CLASS, class DATAPOINT_CLASS, class CUSTOM_UPDATER=DenseSGDUpdater, class CUSTOM_TRAINER=HogwildTrainer>
TrainStatistics RunOnce() {
    // Initialize model and datapoints.
    Model *model;
    std::vector<Datapoint *> datapoints;
    DatasetReader::ReadDataset<MODEL_CLASS, DATAPOINT_CLASS>(FLAGS_data_file, datapoints, model);
    model->SetUp(datapoints);

    // Shuffle the datapoints and assign the order.
    if (FLAGS_shuffle_datapoints) {
      std::random_shuffle(datapoints.begin(), datapoints.end());
    }
    for (int i = 0; i < datapoints.size(); i++) {
	datapoints[i]->SetOrder(i+1);
    }

    // Create updater.
    Updater *updater = NULL;
   	if (FLAGS_dense_svrg) {
	updater = new DenseSVRGUpdater(model, datapoints);
    }
    else if (FLAGS_dense_sgd) {
	updater = new DenseSGDUpdater(model, datapoints);
    }
    else if (FLAGS_batch_grad) {
	updater = new GDUpdater(model, datapoints);
    }
    else {
	updater = new CUSTOM_UPDATER(model, datapoints);
    }

    // Create trainer depending on flag.
    Trainer *trainer = NULL;
	if (FLAGS_hogwild_trainer) {
	trainer = new HogwildTrainer();
    }
	else if (FLAGS_batch_trainer) {
	trainer = new BatchTrainer();
	}
    else {
	trainer = new CUSTOM_TRAINER();
    }

    TrainStatistics stats = trainer->Train(model, datapoints, updater);

    // Delete trainer.
    delete trainer;

    // Delete model and datapoints.
    delete model;
    for_each(datapoints.begin(), datapoints.end(), std::default_delete<Datapoint>());

    // Delete updater.
    delete updater;

    return stats;
}

template<class MODEL_CLASS, class DATAPOINT_CLASS, class CUSTOM_UPDATER=DenseSGDUpdater, class CUSTOM_TRAINER=HogwildTrainer>
void Run() {
    TrainStatistics stats = RunOnce<MODEL_CLASS, DATAPOINT_CLASS, CUSTOM_UPDATER, CUSTOM_TRAINER>();
}
