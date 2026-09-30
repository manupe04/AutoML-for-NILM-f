from __future__ import print_function, division
import time
import sys, os
#sys.path.append(os.path.abspath('../bayesian_optimization/'))
sys.path.append(os.path.abspath('./bayesian_optimization/'))
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from algorithms.GRU.grudisaggregator import GRUDisaggregator
from utils import metrics
import argparse
import json
import pandas as pd


def gru(dataset_path, train_building, train_start, train_end, val_building, val_start, val_end, test_building, test_start, test_end, meter_key, sample_period, num_epochs, patience, optimizer, learning_rate, loss):

    # Start tracking time
    start = time.time()

    # Prepare dataset and options
    print("========== OPEN DATASETS ============")
    dataset_path = dataset_path

    train = DataSet(dataset_path)
    val = DataSet(dataset_path)
    test = DataSet(dataset_path)

    train.set_window(start=train_start, end=train_end)
    val.set_window(start=val_start, end=val_end)
    test.set_window(start=test_start, end=test_end)
    train_building = train_building
    test_building = test_building
    meter_key = meter_key

    sample_period = sample_period

    train_elec = train.buildings[train_building].elec
    val_elec = val.buildings[val_building].elec
    test_elec = test.buildings[test_building].elec

    train_meter = train_elec.submeters()[meter_key]
    val_meter = val_elec.submeters()[meter_key]
    test_meter = test_elec.submeters()[meter_key]

    train_mains = train_elec.mains()
    val_mains = val_elec.mains()
    test_mains = test_elec.mains()

    gru = GRUDisaggregator(patience, optimizer, learning_rate, loss)

    start = time.time()
    print("========== TRAIN ============")
    gru.train(train_mains, train_meter, epochs=num_epochs, sample_period=sample_period)
    # Get number of earlystop epochs
    num_epochs = gru.stopped_epoch if gru.stopped_epoch != 0 else num_epochs
    #gru.export_model("results/gru-model-{}-{}epochs.h5".format(meter_key, num_epochs))
    end = time.time()
    print("Train =", end - start, "seconds.")

    print("========== DISAGGREGATE ============")
    # Validation
    val_disag_filename = 'disag-out-val.h5'
    output = HDFDataStore(val_disag_filename, 'w')
    gru.disaggregate(val_mains, output, train_meter, sample_period=sample_period)
    output.close()
    # Test
    test_disag_filename = 'disag-out-test.h5'
    output = HDFDataStore(test_disag_filename, 'w')
    gru.disaggregate(test_mains, output, train_meter, sample_period=sample_period)
    output.close()

    print("========== RESULTS ============")
    # Validation
    result_val = DataSet(val_disag_filename)
    res_elec_val = result_val.buildings[val_building].elec
    rpaf_val = metrics.recall_precision_accuracy_f1(res_elec_val[meter_key], val_elec[meter_key])

    val_metrics_results_dict = {
        'recall_score': rpaf_val[0],
        'precision_score': rpaf_val[1],
        'accuracy_score': rpaf_val[2],
        'f1_score': rpaf_val[3],
        'mean_absolute_error': metrics.mean_absolute_error(res_elec_val[meter_key], val_elec[meter_key]),
        'mean_squared_error': metrics.mean_square_error(res_elec_val[meter_key], val_elec[meter_key]),
        'relative_error_in_total_energy': metrics.relative_error_total_energy(res_elec_val[meter_key],
                                                                              val_elec[meter_key]),
        'nad': metrics.nad(res_elec_val[meter_key], val_elec[meter_key]),
        'sae': metrics.sae(res_elec_val[meter_key], val_elec[meter_key]),
        'disaggregation_accuracy': metrics.disaggregation_accuracy(res_elec_val[meter_key], val_elec[meter_key])
    }

    # Test
    result = DataSet(test_disag_filename)
    res_elec = result.buildings[test_building].elec
    rpaf = metrics.recall_precision_accuracy_f1(res_elec[meter_key], test_elec[meter_key])

    test_metrics_results_dict = {
        'recall_score': rpaf[0],
        'precision_score': rpaf[1],
        'accuracy_score': rpaf[2],
        'f1_score': rpaf[3],
        'mean_absolute_error': metrics.mean_absolute_error(res_elec[meter_key], test_elec[meter_key]),
        'mean_squared_error': metrics.mean_square_error(res_elec[meter_key], test_elec[meter_key]),
        'relative_error_in_total_energy': metrics.relative_error_total_energy(res_elec[meter_key],
                                                                              test_elec[meter_key]),
        'nad': metrics.nad(res_elec[meter_key], test_elec[meter_key]),
        'sae': metrics.sae(res_elec[meter_key], test_elec[meter_key]),
        'disaggregation_accuracy': metrics.disaggregation_accuracy(res_elec[meter_key], test_elec[meter_key])
    }

    # end tracking time
    end = time.time()
    time_taken = end - start  # in seconds
    model_result_data = {
        'val_metrics': val_metrics_results_dict,
        'test_metrics': test_metrics_results_dict,
        'time_taken': format(time_taken, '.2f'),
        'epochs': num_epochs,
    }
    # Close digag_filename
    result.store.close()
    result_val.store.close()

    # Close Dataset files
    train.store.close()
    val.store.close()
    test.store.close()

    return model_result_data

