from nilmtk.electric import align_two_meters
import numpy as np

import math

def tp_tn_fp_fn(states_pred, states_ground):
    tp = np.sum(np.logical_and(states_pred == 1, states_ground == 1))
    fp = np.sum(np.logical_and(states_pred == 1, states_ground == 0))
    fn = np.sum(np.logical_and(states_pred == 0, states_ground == 1))
    tn = np.sum(np.logical_and(states_pred == 0, states_ground == 0))
    return tp, tn, fp, fn

def recall_precision_accuracy_f1(pred, ground):
    aligned_meters = align_two_meters(pred, ground)

    threshold = ground.on_power_threshold()
    chunk_results = []
    sum_samples = 0.0
    for chunk in aligned_meters:
        sum_samples += len(chunk)
        pr = np.array([0 if (p)<threshold else 1 for p in chunk.iloc[:,0]])
        gr = np.array([0 if p<threshold else 1 for p in chunk.iloc[:,1]])

        tp, tn, fp, fn = tp_tn_fp_fn(pr,gr)
        p = sum(pr)
        n = len(pr) - p

        chunk_results.append([tp,tn,fp,fn,p,n])

    if sum_samples == 0:
        return None
    else:
        [tp,tn,fp,fn,p,n] = np.sum(chunk_results, axis=0)

        res_recall = recall(tp,fn)
        res_precision = precision(tp,fp)
        res_f1 = f1(res_precision,res_recall)
        res_accuracy = accuracy(tp,tn,p,n)

        # If value is NaN return None instead so JSON is valid
        res_recall = None if math.isnan(res_recall) else res_recall
        res_precision = None if math.isnan(res_precision) else res_precision
        res_f1 = None if math.isnan(res_f1) else res_f1
        res_accuracy = None if math.isnan(res_accuracy) else res_accuracy

        return (res_recall,res_precision,res_accuracy,res_f1)

def relative_error_total_energy(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    chunk_results = []
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        sum_samples += len(chunk)
        E_pred = sum(chunk.iloc[:,0])
        E_ground = sum(chunk.iloc[:,1])

        chunk_results.append([
                            E_pred,
                            E_ground
                            ])
    if sum_samples == 0:
        return None
    else:
        [E_pred, E_ground] = np.sum(chunk_results,axis=0)
        return abs(E_pred - E_ground) / float(max(E_pred,E_ground))

# Signal Aggregate Error (RNF-02). Usa E_ground como denominador, a
# diferencia de relative_error_total_energy (que usa max(E_pred, E_ground)):
# es la definicion estandar en NILM (Bonfigli et al., 2017).
def sae(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    chunk_results = []
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        sum_samples += len(chunk)
        E_pred = sum(chunk.iloc[:,0])
        E_ground = sum(chunk.iloc[:,1])
        chunk_results.append([E_pred, E_ground])
    if sum_samples == 0:
        return None
    [E_pred, E_ground] = np.sum(chunk_results, axis=0)
    if E_ground == 0:
        return None
    return abs(E_pred - E_ground) / float(E_ground)

# Coeficiente de determinacion: 1 - SSE / SST, con SST la variabilidad de la
# serie real alrededor de su media. 1 es perfecto, 0 equivale a predecir la
# media y puede ser negativo. Se acumula por chunk (sumas) para no cargar todo.
def r2(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    n = 0.0
    sse = suma_y = suma_y2 = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        p, y = chunk.iloc[:, 0].values, chunk.iloc[:, 1].values
        n += len(chunk)
        sse += float(((p - y) ** 2).sum())
        suma_y += float(y.sum())
        suma_y2 += float((y ** 2).sum())
    if n == 0:
        return None
    sst = suma_y2 - suma_y ** 2 / n
    if sst <= 0:
        return None
    return 1 - sse / sst

# Correlacion de Pearson entre prediccion y serie real: mide si suben y bajan
# juntas, sin importar escala ni sesgo (predecir el doble da r = 1). No es la
# raiz de r2: r2 tambien penaliza el error de magnitud y puede ser negativo.
def pearson(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    n = sp = sy = spp = syy = spy = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        p, y = chunk.iloc[:, 0].values, chunk.iloc[:, 1].values
        n += len(chunk)
        sp += float(p.sum()); sy += float(y.sum())
        spp += float((p * p).sum()); syy += float((y * y).sum()); spy += float((p * y).sum())
    if n == 0:
        return None
    cov = spy - sp * sy / n
    vp, vy = spp - sp ** 2 / n, syy - sy ** 2 / n
    if vp <= 0 or vy <= 0:
        return None
    return cov / (vp * vy) ** 0.5

def mean_absolute_error(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    total_sum = 0.0
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        sum_samples += len(chunk)
        total_sum += sum(abs((chunk.iloc[:,0]) - chunk.iloc[:,1]))
    if sum_samples == 0:
        return None
    else:
        return total_sum / sum_samples


def recall(tp,fn):
    return tp/float(tp+fn)

def precision(tp,fp):
    return tp/float(tp+fp)

def f1(prec,rec):
    return 2 * (prec*rec) / float(prec+rec)

def accuracy(tp, tn, p, n):
    return (tp + tn) / float(p + n)

# Normalized Aboluste Distance
def nad(pred, ground):
    aligned_meters = align_two_meters(pred, ground)

    nominator = 0.0
    denominator = 0.0
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        sum_samples += len(chunk)

        nominator += sum(abs((chunk.iloc[:,0]) - chunk.iloc[:,1]))
        denominator += sum(abs(chunk.iloc[:,1]))
    if sum_samples == 0:
        return None
    else:
        return np.sqrt(nominator / denominator)

# Mean Squared Error  np.mean(np.square(y_predict - y))
def mean_square_error(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    total_sum = 0.0
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        sum_samples += len(chunk)
        total_sum += sum(np.square((chunk.iloc[:,0]) - chunk.iloc[:,1]))
    if sum_samples == 0:
        return None
    else:
        return total_sum / sum_samples

# TODO: add in code and edit JSON format
# Lungu's Disaggregation Accuracy for Appliance
def disaggregation_accuracy(pred, ground):

    aligned_meters = align_two_meters(pred, ground)

    nominator = 0.0
    denominator = 0.0
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        sum_samples += len(chunk)

        nominator += np.linalg.norm(chunk.iloc[:,0] - chunk.iloc[:,1], ord=1)
        denominator += np.linalg.norm(chunk.iloc[:,0], ord=1)
    if sum_samples == 0:
        return None
    else:
        return  1 - (float(nominator) / (2*denominator))
    #     denominator += 2 * np.linalg.norm(chunk.iloc[:,0], ord=1)
    # if sum_samples == 0:
    #     return None
    # else:
    #     return  1 - (float(nominator) / denominator)
def calculate_metrics(pred, ground):
    return {
        'mae': mean_absolute_error(pred, ground),
        'mse': mean_square_error(pred, ground),
        'nad': nad(pred, ground),
        'rel_error': relative_error_total_energy(pred, ground),
        'disagg_acc': disaggregation_accuracy(pred, ground),
        'precision_recall_acc_f1': recall_precision_accuracy_f1(pred, ground)
    }
