"""Corte temprano comun para los desagregadores de Keras.

Antes solo FCNN usaba `patience`: el resto entrenaba siempre todas las epocas.
La validacion es el ultimo 10% del tramo de entrenamiento (Keras toma
`validation_split` del final, antes de mezclar), o sea un corte temporal.
"""
from tensorflow.keras.callbacks import EarlyStopping

VALIDATION_SPLIT = 0.1


def fit_con_corte(model, X, Y, epochs, batch_size, patience):
    """Entrena con corte temprano y devuelve la cantidad de epocas corridas."""
    callbacks = []
    if patience:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=patience,
                                       restore_best_weights=True))
    hist = model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=True,
                     validation_split=VALIDATION_SPLIT, callbacks=callbacks)
    return len(hist.history['loss'])
