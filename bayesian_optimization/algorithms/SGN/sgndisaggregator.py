from __future__ import print_function, division

import numpy as np
import pandas as pd
import h5py
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input, Multiply
from nilmtk.legacy.disaggregate import Disaggregator
from algorithms.multi_casa import EntrenamientoMultiCasa


class SGNDisaggregator(EntrenamientoMultiCasa, Disaggregator):
    """Subtask Gated Network (Shin et al., 2019).

    Dos subredes paralelas sobre la misma ventana de la senal agregada:

      - Subred de regresion: estima la potencia del electrodomestico.
      - Subred de clasificacion: estima el estado on/off (salida sigmoide).

    La salida final es el producto de ambas. La subred de clasificacion
    actua como compuerta (gate): cuando estima que el electrodomestico
    esta apagado, anula la estimacion de potencia. Eso reduce los errores
    de estado, que son los que mas afectan la confianza del usuario final.

    La interfaz es identica a la de Seq2PointDisaggregator para que el
    framework AutoML4NILM pueda usarla sin cambios.
    """

    def __init__(self, patience, optimizer, learning_rate, loss, window_size=99):
        self.MODEL_NAME = "SGN"
        self.mmax = None
        self.patience = patience
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.window_size = window_size
        self.stopped_epoch = 0
        self.model = self._create_model(self.optimizer, self.learning_rate, self.loss)

    def _conv_stack(self, x, prefix):
        """Pila convolucional compartida por ambas subredes.

        Replica la arquitectura de Seq2Point para que la comparacion entre
        modelos aisle el efecto de la compuerta y no el de la profundidad.
        """
        x = Conv1D(30, 10, activation="relu", padding="same", name=prefix + "_conv1")(x)
        x = Conv1D(30, 8, activation="relu", padding="same", name=prefix + "_conv2")(x)
        x = Conv1D(40, 6, activation="relu", padding="same", name=prefix + "_conv3")(x)
        x = Conv1D(50, 5, activation="relu", padding="same", name=prefix + "_conv4")(x)
        x = Conv1D(50, 5, activation="relu", padding="same", name=prefix + "_conv5")(x)
        x = Flatten(name=prefix + "_flat")(x)
        x = Dense(1024, activation="relu", name=prefix + "_dense")(x)
        return x

    def _create_model(self, optimizer, learning_rate, loss):
        inputs = Input(shape=(self.window_size, 1), name="aggregate_input")

        # Subred de regresion: cuanta potencia consume.
        reg = self._conv_stack(inputs, "reg")
        power = Dense(1, activation="linear", name="power_output")(reg)

        # Subred de clasificacion: esta encendido o apagado.
        cls = self._conv_stack(inputs, "cls")
        state = Dense(1, activation="sigmoid", name="state_output")(cls)

        # Compuerta: la estimacion de potencia se anula si el estado es apagado.
        gated = Multiply(name="gated_output")([power, state])

        model = Model(inputs=inputs, outputs=gated, name="SGN")
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

    def _ventanas_entrenamiento(self, mains, aparato):
        # train() y train_casas() vienen de EntrenamientoMultiCasa
        X, _ = self._create_windows(mains)
        return X, np.asarray(aparato)

    def disaggregate_chunk(self, mains):
        mains.fillna(0, inplace=True)
        normalized = self._nx(mains)
        X, index = self._create_windows(normalized)

        predictions = self.model.predict(X, batch_size=128)
        predictions = self._dy(predictions.flatten())

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
        # compile=False: Keras 3 no puede deserializar la loss compilada
        # ('mse') guardada en el formato legacy H5, y load_model() revienta
        # aunque el modelo en si este intacto. No hace falta recompilar para
        # disaggregate(), que solo usa model.predict().
        self.model = load_model(filename, compile=False)
        with h5py.File(filename, 'r') as hf:
            self.mmax = np.array(hf.get('disaggregator-data').get('mmax'))[0]
            self.window_size = int(np.array(hf.get('disaggregator-data').get('window_size'))[0])
            self._leer_normalizacion(hf.get('disaggregator-data'))

    def export_model(self, filename):
        self.model.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('mmax', data=[self.mmax])
            gr.create_dataset('window_size', data=[self.window_size])
            self._guardar_normalizacion(gr)
