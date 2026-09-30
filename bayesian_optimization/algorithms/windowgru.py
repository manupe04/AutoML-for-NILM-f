from __future__ import print_function, division
import time
import sys, os
sys.path.append(os.path.abspath('./bayesian_optimization/'))

from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from nilmtk import DataSet, HDFDataStore
from algorithms.WindowGRU.windowGRUdisaggregator import WindowGRUDisaggregator
from utils import metrics
import argparse
import json
import pandas as pd


def window_gru(dataset_path, train_building, train_start, train_end, val_building, val_start, val_end,
               test_building, test_start, test_end, meter_key, sample_period, num_epochs, patience,
               optimizer, learning_rate, loss, window_size):

    # Start tracking time
    start = time.time()

    print("========== OPEN DATASETS ============")
    train = DataSet(dataset_path)
    val = DataSet(dataset_path)
    test = DataSet(dataset_path)

    train.set_window(start=train_start, end=train_end)
    val.set_window(start=val_start, end=val_end)
    test.set_window(start=test_start, end=test_end)

    train_elec = train.buildings[train_building].elec
    val_elec = val.buildings[val_building].elec
    test_elec = test.buildings[test_building].elec

    train_meter = train_elec.submeters()[meter_key]
    val_meter = val_elec.submeters()[meter_key]
    test_meter = test_elec.submeters()[meter_key]

    train_mains = train_elec.mains()
    val_mains = val_elec.mains()
    test_mains = test_elec.mains()

    model = WindowGRUDisaggregator(patience=patience,
                                   optimizer=optimizer,
                                   learning_rate=learning_rate,
                                   loss=loss,
                                   window_size=window_size)

    print("========== TRAIN ============")
    model.train(train_mains, train_meter, epochs=num_epochs, sample_period=sample_period)
    num_epochs = model.stopped_epoch if model.stopped_epoch != 0 else num_epochs

    print("========== DISAGGREGATE ============")
    # Validation
    val_disag_filename = 'disag-window-val.h5'
    output = HDFDataStore(val_disag_filename, 'w')
    model.disaggregate(val_mains, output, train_meter, sample_period=sample_period)
    output.close()

    # Test
    test_disag_filename = 'disag-window-test.h5'
    output = HDFDataStore(test_disag_filename, 'w')
    model.disaggregate(test_mains, output, train_meter, sample_period=sample_period)
    output.close()

    print("========== RESULTS ============")
    # Validation
    result_val = DataSet(val_disag_filename)
    res_elec_val = result_val.buildings[val_building].elec
    rpaf_val = metrics.recall_precision_accuracy_f1(res_elec_val[meter_key], val_elec[meter_key])

    val_metrics = {
        'recall_score': rpaf_val[0],
        'precision_score': rpaf_val[1],
        'accuracy_score': rpaf_val[2],
        'f1_score': rpaf_val[3],
        'mean_absolute_error': metrics.mean_absolute_error(res_elec_val[meter_key], val_elec[meter_key]),
        'mean_squared_error': metrics.mean_square_error(res_elec_val[meter_key], val_elec[meter_key]),
        'relative_error_in_total_energy': metrics.relative_error_total_energy(res_elec_val[meter_key], val_elec[meter_key]),
        'nad': metrics.nad(res_elec_val[meter_key], val_elec[meter_key]),
        'sae': metrics.sae(res_elec_val[meter_key], val_elec[meter_key]),
        'disaggregation_accuracy': metrics.disaggregation_accuracy(res_elec_val[meter_key], val_elec[meter_key])
    }

    # Test
    result = DataSet(test_disag_filename)
    res_elec = result.buildings[test_building].elec
    rpaf = metrics.recall_precision_accuracy_f1(res_elec[meter_key], test_elec[meter_key])

    test_metrics = {
        'recall_score': rpaf[0],
        'precision_score': rpaf[1],
        'accuracy_score': rpaf[2],
        'f1_score': rpaf[3],
        'mean_absolute_error': metrics.mean_absolute_error(res_elec[meter_key], test_elec[meter_key]),
        'mean_squared_error': metrics.mean_square_error(res_elec[meter_key], test_elec[meter_key]),
        'relative_error_in_total_energy': metrics.relative_error_total_energy(res_elec[meter_key], test_elec[meter_key]),
        'nad': metrics.nad(res_elec[meter_key], test_elec[meter_key]),
        'sae': metrics.sae(res_elec[meter_key], test_elec[meter_key]),
        'disaggregation_accuracy': metrics.disaggregation_accuracy(res_elec[meter_key], test_elec[meter_key])
    }

    # Time tracking
    time_taken = time.time() - start

    result_dict = {
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'time_taken': format(time_taken, '.2f'),
        'epochs': num_epochs,
    }

    result_val.store.close()
    result.store.close()
    train.store.close()
    val.store.close()
    test.store.close()

    return result_dict


def main():
#
#     # Take in arguments from command line
     parser = argparse.ArgumentParser(description='GRU')
     parser.add_argument('--datapath', '-d', type=str,  required=True,
                         help='hd5 filepath')

     parser.add_argument('--train_building', type=int, required=True)
     parser.add_argument('--train_start', type=str, default=None, help='YYYY-MM-DD')
     parser.add_argument('--train_end', type=str, required=True, help='YYYY-MM-DD')
     parser.add_argument('--val_building', type=int, required=True)
     parser.add_argument('--val_start', type=str, default=None, help='YYYY-MM-DD')
     parser.add_argument('--val_end', type=str, required=True, help='YYYY-MM-DD')
#
     parser.add_argument('--test_building', type=int, required=True)
     parser.add_argument('--test_start', type=str, required=True, help='YYYY-MM-DD')
     parser.add_argument('--test_end', type=str, default=None, help='YYYY-MM-DD')
#
     parser.add_argument('--appliance', type=str, required=True)
     parser.add_argument('--sampling_rate', type=int, required=True)
#
#     # Model specific options and hyperparameters
     parser.add_argument('--epochs', type=int, default=50)
     parser.add_argument('--window_size', type=int, default=60)


     parser.add_argument('--learning_rate', type=float, required=True)
     parser.add_argument('--patience', type=int, default=10)
     parser.add_argument('--optimizer', type=str, required=True)

     parser.add_argument('--loss', type=str,required=True)

     args = parser.parse_args()
#meter_key, sample_period, num_epochs, patience, sequence_length, optimizer, learning_rate, loss
     hd5_filepath = args.datapath
     train_building = args.train_building
     train_start = pd.Timestamp(args.train_start) if args.train_start != None else None
     train_end = pd.Timestamp(args.train_end)
     val_building = args.val_building
     val_start = pd.Timestamp(args.val_start)
     val_end = pd.Timestamp(args.val_end)
     test_building = args.test_building
     test_start = pd.Timestamp(args.test_start)
     test_end = pd.Timestamp(args.test_end) if args.test_end != None else None
     appliance = args.appliance
     downsampling_period = args.sampling_rate
     epochs = args.epochs

     learning_rate=args.learning_rate
     optimizer= args.optimizer
     loss=args.loss
     patience=args.patience
     window_size=args.window_size


     model_result_data = window_gru(
         dataset_path=hd5_filepath,
         train_building=train_building, train_start=train_start, train_end=train_end,
         val_building=val_building, val_start=val_start, val_end=val_end,
         test_building=test_building, test_start=test_start, test_end=test_end,
         meter_key=appliance,
         sample_period=downsampling_period,
         num_epochs=epochs, learning_rate=learning_rate,optimizer=optimizer,loss=loss,patience=patience, window_size=window_size
)

    # Write options and results to file
     with open('windowgru_json.json', 'a+') as outfile:
         json.dump(model_result_data, outfile, sort_keys=True,
                  indent=4, separators=(',', ': '))
     print(model_result_data)
#
if __name__ == "__main__":
     main()
