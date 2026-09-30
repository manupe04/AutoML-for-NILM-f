from __future__ import print_function, division
import sys
sys.path.append('/content/drive/MyDrive/automl4nilm2')

import numpy as np
import pandas as pd
import h5py
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from nilmtk.legacy.disaggregate import Disaggregator
from algorithms.corte_temprano import fit_con_corte

class Seq2PointDisaggregator(Disaggregator):
    def __init__(self, patience, optimizer, learning_rate, loss, window_size=99):
        self.MODEL_NAME = "Seq2Point"
        self.mmax = None
        self.patience = patience
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.window_size = window_size
        self.stopped_epoch = 0
        self.model = self._create_model(self.optimizer, self.learning_rate, self.loss)

    def _create_model(self, optimizer, learning_rate, loss):
        model = Sequential()
        model.add(Conv1D(30, 10, activation="relu", padding="same", input_shape=(self.window_size, 1)))
        model.add(Conv1D(30, 8, activation="relu", padding="same"))
        model.add(Conv1D(40, 6, activation="relu", padding="same"))
        model.add(Conv1D(50, 5, activation="relu", padding="same"))
        model.add(Conv1D(50, 5, activation="relu", padding="same"))
        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def _normalize(self, chunk, mmax):
        return chunk / mmax

    def _denormalize(self, chunk, mmax):
        return chunk * mmax

    def _create_windows(self, series):
        pad = self.window_size // 2
        padded = np.pad(series, (pad, pad), mode='constant')
        X, idx = [], []
        for i in range(len(series)):
            window = padded[i:i + self.window_size]
            X.append(window)
            idx.append(series.index[i])
        return np.array(X).reshape(-1, self.window_size, 1), pd.Index(idx)

    def train(self, mains, meter, epochs=1, batch_size=128, **load_kwargs):
        main_series = mains.power_series(**load_kwargs)
        meter_series = meter.power_series(**load_kwargs)

        run = True
        mainchunk = next(main_series)
        meterchunk = next(meter_series)
        if self.mmax is None:
            self.mmax = mainchunk.max()

        while run:
            mainchunk = self._normalize(mainchunk, self.mmax)
            meterchunk = self._normalize(meterchunk, self.mmax)
            self.train_on_chunk(mainchunk, meterchunk, epochs, batch_size)
            try:
                mainchunk = next(main_series)
                meterchunk = next(meter_series)
            except:
                run = False

    def train_on_chunk(self, mainchunk, meterchunk, epochs, batch_size):
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)

        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = mainchunk[ix]
        meterchunk = meterchunk[ix]

        X, idx = self._create_windows(mainchunk)
        Y = np.array(meterchunk)

        self.stopped_epoch = max(self.stopped_epoch, fit_con_corte(self.model, X, Y, epochs, batch_size, self.patience))

    def disaggregate_chunk(self, mains):
        mains.fillna(0, inplace=True)
        normalized = self._normalize(mains, self.mmax)
        X, index = self._create_windows(normalized)

        predictions = self.model.predict(X, batch_size=128)
        predictions = self._denormalize(predictions.flatten(), self.mmax)

        return pd.DataFrame({0: predictions}, index=index)

    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):
        load_kwargs = self._pre_disaggregation_checks(load_kwargs)
        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = f'/building{mains.building()}'
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < 100:
                continue
            print("New sensible chunk: {}".format(len(chunk)))

            timeframes.append(chunk.timeframe)
            measurement = chunk.name

            appliance_power = self.disaggregate_chunk(chunk)
            appliance_power[appliance_power < 0] = 0

            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = meter_metadata.instance()
            df = pd.DataFrame(appliance_power.values, index=appliance_power.index, columns=cols, dtype="float32")
            key = f'{building_path}/elec/meter{meter_instance}'
            output_datastore.append(key, df)

            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_datastore.append(key=mains_data_location, value=mains_df)

        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter_metadata]
            )

    def import_model(self, filename):
        self.model = load_model(filename, compile=False)
        with h5py.File(filename, 'r') as hf:
            self.mmax = np.array(hf.get('disaggregator-data').get('mmax'))[0]
            self.window_size = int(np.array(hf.get('disaggregator-data').get('window_size'))[0])

    def export_model(self, filename):
        self.model.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('mmax', data=[self.mmax])
            gr.create_dataset('window_size', data=[self.window_size])
