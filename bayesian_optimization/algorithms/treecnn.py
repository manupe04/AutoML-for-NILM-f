import time
import sys, os
sys.path.append(os.path.abspath('./bayesian_optimization/'))
from nilmtk import DataSet, HDFDataStore
from algorithms.TREECNN.treecnndisaggregator import TreeCNNDisaggregator
from algorithms.multi_casa import abrir_casas_extra
from utils import metrics
import argparse
import json
import pandas as pd


def treecnn(dataset_path, train_building, train_start, train_end, val_building, val_start, val_end,
            test_building, test_start, test_end, meter_key, sample_period, num_epochs, patience,
            optimizer, learning_rate, loss, window_size, kernel_size=7,
            train_extra=None, normalizacion="pico", potencia_aparato=None,
            exportar=None):
    """Wrapper que integra TreeCNN a AutoML4NILM, con la misma firma que sgn().

    Usa el camino train()/disaggregate() de TreeCNNDisaggregator (un solo
    electrodomestico por corrida), que es el compatible con este pipeline.
    El camino train_multi() -- la cascada real de TreeCNN -- queda fuera de
    la optimizacion bayesiana porque esta entrena varios aparatos a la vez y
    el framework evalua un aparato por corrida.
    """

    start = time.time()

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

    model = TreeCNNDisaggregator(patience=patience,
                                  optimizer=optimizer,
                                  learning_rate=learning_rate,
                                  loss=loss,
                                  window_size=window_size,
                                  kernel_size=kernel_size)

    # train_extra: casas de train adicionales [(building, inicio, fin), ...]
    model.configurar_normalizacion(normalizacion, potencia_aparato)
    pares_extra, abiertos = abrir_casas_extra(dataset_path, train_extra, meter_key)
    model.train_casas([(train_mains, train_meter)] + pares_extra, epochs=num_epochs, sample_period=sample_period)
    for ds in abiertos:
        ds.store.close()
    if exportar:
        # Modelo entrenado (con su normalizacion) para evaluar_teca.py / publicar
        model.export_model(exportar)
    num_epochs = model.stopped_epoch if model.stopped_epoch != 0 else num_epochs

    val_disag_filename = 'disag-TreeCNN-val.h5'
    output = HDFDataStore(val_disag_filename, 'w')
    model.disaggregate(val_mains, output, train_meter, sample_period=sample_period)
    output.close()

    test_disag_filename = 'disag-TreeCNN-test.h5'
    output = HDFDataStore(test_disag_filename, 'w')
    model.disaggregate(test_mains, output, train_meter, sample_period=sample_period)
    output.close()

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
        'sae': metrics.sae(res_elec_val[meter_key], val_elec[meter_key]),
        'r2': metrics.r2(res_elec_val[meter_key], val_elec[meter_key]),
        'pearson': metrics.pearson(res_elec_val[meter_key], val_elec[meter_key]),
        'nad': metrics.nad(res_elec_val[meter_key], val_elec[meter_key]),
        'disaggregation_accuracy': metrics.disaggregation_accuracy(res_elec_val[meter_key], val_elec[meter_key])
    }

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
        'sae': metrics.sae(res_elec[meter_key], test_elec[meter_key]),
        'r2': metrics.r2(res_elec[meter_key], test_elec[meter_key]),
        'pearson': metrics.pearson(res_elec[meter_key], test_elec[meter_key]),
        'nad': metrics.nad(res_elec[meter_key], test_elec[meter_key]),
        'disaggregation_accuracy': metrics.disaggregation_accuracy(res_elec[meter_key], test_elec[meter_key])
    }

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
    parser = argparse.ArgumentParser(description='TreeCNN Disaggregator')
    parser.add_argument('--datapath', '-d', type=str, required=True)

    parser.add_argument('--train_building', type=int, required=True)
    parser.add_argument('--train_start', type=str, default=None)
    parser.add_argument('--train_end', type=str, required=True)
    parser.add_argument('--val_building', type=int, required=True)
    parser.add_argument('--val_start', type=str)
    parser.add_argument('--val_end', type=str, required=True)

    parser.add_argument('--test_building', type=int, required=True)
    parser.add_argument('--test_start', type=str, required=True)
    parser.add_argument('--test_end', type=str, default=None)

    parser.add_argument('--appliance', type=str, required=True)
    parser.add_argument('--sampling_rate', type=int, required=True)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--window_size', type=int, default=128)

    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--optimizer', type=str, required=True)
    parser.add_argument('--loss', type=str, required=True)

    args = parser.parse_args()

    model_result_data = treecnn(
        dataset_path=args.datapath,
        train_building=args.train_building, train_start=pd.Timestamp(args.train_start) if args.train_start else None, train_end=pd.Timestamp(args.train_end),
        val_building=args.val_building, val_start=pd.Timestamp(args.val_start), val_end=pd.Timestamp(args.val_end),
        test_building=args.test_building, test_start=pd.Timestamp(args.test_start), test_end=pd.Timestamp(args.test_end) if args.test_end else None,
        meter_key=args.appliance, sample_period=args.sampling_rate,
        num_epochs=args.epochs, learning_rate=args.learning_rate, optimizer=args.optimizer,
        loss=args.loss, patience=args.patience, window_size=args.window_size
    )

    with open('treecnn_json.json', 'a+') as outfile:
        json.dump(model_result_data, outfile, sort_keys=True, indent=4, separators=(',', ': '))
    print(model_result_data)


if __name__ == "__main__":
    main()
