#!/bin/env/python

import functools
import numpy as np
import pprint
import scipy
import time

from . import amm
from . import matmul_datasets as md
from . import pyience as pyn
from . import compress

from . import amm_methods
from . import amm_methods as methods

from joblib import Memory
_memory = Memory('.', verbose=0)


# NUM_TRIALS = 1
NUM_TRIALS = 10


# @_memory.cache
def _estimator_for_method_id(method_id, **method_hparams):
    return methods.METHOD_TO_ESTIMATOR[method_id](**method_hparams)


def _hparams_for_method(method_id):
    if method_id in methods.SKETCH_METHODS:
        # dvals = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]  # d=1 undef on fd methods
        # dvals = [1, 2, 4, 8, 16, 32, 64, 128]
        dvals = [1, 2, 4, 8, 16, 32, 64]
        # dvals = [1, 2, 4, 8, 16, 32]
        # dvals = [1, 2, 4, 8]
        # dvals = [32] # TODO rm after debug
        # dvals = [16] # TODO rm after debug
        # dvals = [8] # TODO rm after debug
        # dvals = [4] # TODO rm after debug
        # dvals = [3] # TODO rm after debug
        # dvals = [2] # TODO rm after debug
        # dvals = [1] # TODO rm after debug
        if method_id == methods.METHOD_SPARSE_PCA:
            # first one gets it to not return all zeros on caltech
            alpha_vals = (1. / 16384, .03125, .0625, .125, .25, .5, 1, 2, 4, 8)
            # alpha_vals = (.0625, .125, .25, .5, 1, 2, 4, 8)
            # alpha_vals = (.0625, .125)
            # alpha_vals = [.0625] # TODO rm
            # alpha_vals = [.03125] # TODO rm
            # alpha_vals = [1./1024] # TODO rm
            # alpha_vals = [1./16384] # TODO rm
            # alpha_vals = [0] # TODO rm
            # alpha_vals = (2, 4, 5)
            # alpha_vals = [.1]
            # alpha_vals = [1.]
            # alpha_vals = [10.]
            # alpha_vals = [20.]
            # alpha_vals = [50.]
            return [{'d': d, 'alpha': alpha}
                    for d in dvals for alpha in alpha_vals]
        return [{'d': dval} for dval in dvals]

    if method_id in methods.VQ_METHODS:
        # mvals = [1, 2, 4, 8, 16, 32, 64]
        mvals = [2, 4, 8, 16, 32, 64]
        # mvals = [64]
        # mvals = [1, 2, 4, 8, 16]
        # mvals = [1, 2, 4, 8]
        # mvals = [8, 16] # TODO rm after debug
        # mvals = [8, 16, 64] # TODO rm after debug
        # mvals = [128] # TODO rm after debug
        # mvals = [64] # TODO rm after debug
        # mvals = [32] # TODO rm after debug
        # mvals = [16] # TODO rm after debug
        # mvals = [8] # TODO rm after debug
        # mvals = [4] # TODO rm after debug
        # mvals = [1] # TODO rm after debug

        if method_id == methods.METHOD_MITHRAL:
            lut_work_consts = (2, 4, -1)
            # lut_work_consts = [-1] # TODO rm
            params = []
            for m in mvals:
                for const in lut_work_consts:
                    params.append({'ncodebooks': m, 'lut_work_const': const})
            return params
            # return [{'ncodebooks': 16, 'lut_work_const': 2}] # TODO lut_work_const? nnz? 

        return [{'ncodebooks': m} for m in mvals]
    
    if method_id in [methods.METHOD_EXACT, methods.METHOD_SCALAR_QUANTIZE]:
        return [{}]
    
    if method_id == methods.METHOD_INTELLISTREAM_AMM: # parameter is set in config.csv, not here, just give a dummy value to make sure it does not affect the original codebase
        return [{'ncodebooks': 16, 'lut_work_const': 2}]

    raise ValueError(f"Unrecognized method: '{method_id}'")


def _ntrials_for_method(method_id, ntasks):
    # return 1 # TODO rm
    if ntasks > 1:  # no need to avg over trials if avging over multiple tasks
        return 1
    # return NUM_TRIALS if method_id in methods.NONDETERMINISTIC_METHODS else 1
    return NUM_TRIALS if method_id in methods.RANDOM_SKETCHING_METHODS else 1


# ================================================================ metrics

def _compute_compression_metrics(ar):
    # if quantize_to_type is not None:
    #     ar = ar.astype(quantize_to_type)
    # ar -= np.min(ar)
    # ar /= (np.max(ar) / 65535)  # 16 bits
    # ar -= 32768  # center at 0
    # ar = ar.astype(np.int16)

    # elem_sz = ar.dtype.itemsize
    # return {'nbytes_raw': ar.nbytes,
    #         'nbytes_blosc_noshuf': len(_blosc_compress(
    #             ar, elem_sz=elem_sz, shuffle=blosc.NOSHUFFLE)),
    #         'nbytes_blosc_byteshuf': len(_blosc_compress(
    #             ar, elem_sz=elem_sz, shuffle=blosc.SHUFFLE)),
    #         'nbytes_blosc_bitshuf': len(_blosc_compress(
    #             ar, elem_sz=elem_sz, shuffle=blosc.BITSHUFFLE)),
    #         'nbytes_zstd': len(_zstd_compress(ar)),
    #         'nbits_cost': nbits_cost(ar).sum() // 8,
    #         'nbits_cost_zigzag':
    #             nbits_cost(zigzag_encode(ar), signed=False).sum() // 8,
    #         'nbytes_sprintz': compress.sprintz_packed_size(ar)
    #         }

    return {'nbytes_raw': ar.nbytes,
            'nbytes_sprintz': compress.sprintz_packed_size(ar)}


def _cossim(Y, Y_hat):
    ynorm = np.linalg.norm(Y) + 1e-20
    yhat_norm = np.linalg.norm(Y_hat) + 1e-20
    return ((Y / ynorm) * (Y_hat / yhat_norm)).sum()


def _compute_metrics(task, Y_hat, compression_metrics=True, **sink):
    Y = task.Y_test
    diffs = Y - Y_hat
    raw_mse = np.mean(diffs * diffs)
    normalized_mse = raw_mse / np.var(Y)
    # Y_meannorm = Y - Y.mean()
    # Y_hat_meannorm = Y_hat - Y_hat.mean()
    # ynorm = np.linalg.norm(Y_meannorm) + 1e-20
    # yhat_norm = np.linalg.norm(Y_hat_meannorm) + 1e-20
    # r = ((Y_meannorm / ynorm) * (Y_hat_meannorm / yhat_norm)).sum()
    metrics = {'raw_mse': raw_mse, 'normalized_mse': normalized_mse,
               'corr': _cossim(Y - Y.mean(), Y_hat - Y_hat.mean()),
               'cossim': _cossim(Y, Y_hat),  # 'bias': diffs.mean(),
               'y_mean': Y.mean(), 'y_std': Y.std(),
               'yhat_std': Y_hat.std(), 'yhat_mean': Y_hat.mean()}
    if compression_metrics:

        # Y_q = compress.quantize(Y, nbits=8)
        # Y_hat_q = compress.quantize(Y_hat, nbits=8)
        # diffs_q = Y_q - Y_hat_q
        # # diffs_q = compress.zigzag_encode(diffs_q).astype(np.uint8)
        # assert Y_q.dtype == np.int8
        # assert diffs_q.dtype == np.int8

        Y_q = compress.quantize(Y, nbits=12)
        Y_hat_q = compress.quantize(Y_hat, nbits=12)
        diffs_q = Y_q - Y_hat_q
        assert Y_q.dtype == np.int16
        assert diffs_q.dtype == np.int16

        # Y_q = quantize_i16(Y)

        # # quantize to 16 bits
        # Y = Y - np.min(Y)
        # Y /= (np.max(Y) / 65535)  # 16 bits
        # Y -= 32768  # center at 0
        # Y = Y.astype(np.int16)
        # diffs =

        metrics_raw = _compute_compression_metrics(Y_q)
        metrics.update({k + '_orig': v for k, v in metrics_raw.items()})
        metrics_raw = _compute_compression_metrics(diffs_q)
        metrics.update({k + '_diffs': v for k, v in metrics_raw.items()})

    if task.info:
        problem = task.info['problem']
        metrics['problem'] = problem
        if problem == 'softmax':
            lbls = task.info['lbls_test'].astype(np.int32) # (10000,)
            b = task.info['biases'] # (10,)
            t = time.perf_counter()
            logits_amm = Y_hat + b # (10000, 10) + (10,)
            logits_orig = Y + b
            lbls_amm = np.argmax(logits_amm, axis=1).astype(np.int32) # (10000,)
            lbls_orig = np.argmax(logits_orig, axis=1).astype(np.int32)
            duration_secs = time.perf_counter() - t
            print(f"+bias, softmax time: {duration_secs}")
            # print("Y_hat shape : ", Y_hat.shape)
            # print("lbls hat shape: ", lbls_amm.shape)
            # print("lbls amm : ", lbls_amm[:20])
            metrics['acc_amm'] = np.mean(lbls_amm == lbls)
            metrics['acc_orig'] = np.mean(lbls_orig == lbls)

        elif problem in ('1nn', 'rbf'):
            lbls = task.info['lbls_test'].astype(np.int32)
            lbls_centroids = task.info['lbls_centroids']
            lbls_hat_1nn = []
            rbf_lbls_hat = []
            W = task.W_test
            centroid_norms_sq = (W * W).sum(axis=0)
            sample_norms_sq = (task.X_test * task.X_test).sum(
                axis=1, keepdims=True)

            k = W.shape[1]
            nclasses = np.max(lbls_centroids) + 1
            affinities = np.zeros((k, nclasses), dtype=np.float32)
            for kk in range(k):
                affinities[kk, lbls_centroids[kk]] = 1

            for prods in [Y_hat, Y]:
                dists_sq_hat = (-2 * prods) + centroid_norms_sq + sample_norms_sq
                # 1nn classification
                centroid_idx = np.argmin(dists_sq_hat, axis=1)
                lbls_hat_1nn.append(lbls_centroids[centroid_idx])
                # rbf kernel classification (bandwidth=1)
                # gamma = 1. / np.sqrt(W.shape[0])
                # gamma = 1. / W.shape[0]
                gamma = 1
                similarities = scipy.special.softmax(-dists_sq_hat * gamma, axis=1)
                class_probs = similarities @ affinities
                rbf_lbls_hat.append(np.argmax(class_probs, axis=1))

            lbls_amm_1nn, lbls_orig_1nn = lbls_hat_1nn
            rbf_lbls_amm, rbf_lbls_orig = rbf_lbls_hat
            metrics['acc_amm_1nn'] = np.mean(lbls_amm_1nn == lbls)
            metrics['acc_orig_1nn'] = np.mean(lbls_orig_1nn == lbls)
            metrics['acc_amm_rbf'] = np.mean(rbf_lbls_amm == lbls)
            metrics['acc_orig_rbf'] = np.mean(rbf_lbls_orig == lbls)

            if problem == '1nn':
                lbls_amm, lbls_orig = rbf_lbls_amm, rbf_lbls_orig
            elif problem == 'rbf':
                lbls_amm, lbls_orig = rbf_lbls_amm, rbf_lbls_orig

            orig_acc_key = 'acc-1nn-raw'
            if orig_acc_key in task.info:
                metrics[orig_acc_key] = task.info[orig_acc_key]

            metrics['acc_amm'] = np.mean(lbls_amm == lbls)
            metrics['acc_orig'] = np.mean(lbls_orig == lbls)
        elif problem == 'sobel':
            assert Y.shape[1] == 2
            grad_mags_true = np.sqrt((Y * Y).sum(axis=1))
            grad_mags_hat = np.sqrt((Y_hat * Y_hat).sum(axis=1))
            diffs = grad_mags_true - grad_mags_hat
            metrics['grad_mags_nmse'] = (
                (diffs * diffs).mean() / grad_mags_true.var())
        elif problem.lower().startswith('dog'):
            # difference of gaussians
            assert Y.shape[1] == 2
            Z = Y[:, 0] - Y[:, 1]
            Z_hat = Y_hat[:, 0] - Y_hat[:, 1]
            diffs = Z - Z_hat
            metrics['dog_nmse'] = (diffs * diffs).mean() / Z.var()

    return metrics


# ================================================================ driver funcs

def _eval_amm(task, est, fixedB=True, **metrics_kwargs):
    est.reset_for_new_task()
    if fixedB:
        est.set_B(task.W_test)

    # print("eval_amm validating task: ", task.name)
    # task.validate(train=False, test=True)
    # print(f"task {task.name} matrix hashes:")
    # pprint.pprint(task._hashes())

    # print("task: ", task.name)
    # print("X_test shape: ", task.X_test.shape)
    # print("W_test shape: ", task.W_test.shape)
    t = time.perf_counter()
    # Y_hat = est.predict(task.X_test.copy(), task.W_test.copy())
    Y_hat = est.predict(task.X_test, task.W_test)
    # Y_hat = task.X_test @ task.W_test  # yep, zero error
    duration_secs = time.perf_counter() - t

    metrics = _compute_metrics(task, Y_hat, **metrics_kwargs)
    metrics['secs'] = duration_secs
    # metrics['nmultiplies'] = est.get_nmuls(task.X_test, task.W_test)
    metrics.update(est.get_speed_metrics(
        task.X_test, task.W_test, fixedB=fixedB))

    # print("eval_amm re-validating task: ", task.name)
    # task.validate(train=False, test=True)
    # print(f"task {task.name} matrix hashes:")
    # pprint.pprint(task.hashes())

    return metrics

def _eval_amm_for_intellistream_amm(task, est, fixedB=True, intellistream_amm_config_load_path=None, 
intellistream_amm_metric_save_path=None):
    est.reset_for_new_task()
    if fixedB:
        est.set_B(task.W_test)
        
    est.intellistream_amm_metric_save_path = intellistream_amm_metric_save_path
    est.intellistream_amm_config_load_path = intellistream_amm_config_load_path

    ##### 1. run AMM #####
    t = time.perf_counter()
    # Y_hat = est.predict(task.X_test.copy(), task.W_test.copy())
    Y_hat = est.predict(task.X_test, task.W_test)
    # Y_hat = task.X_test @ task.W_test  # yep, zero error
    duration_secs = time.perf_counter() - t

    ##### 2. +bias, softmax, accuracy #####
    Y = task.Y_test
    diffs = Y - Y_hat
    raw_mse = np.mean(diffs * diffs)
    normalized_mse = raw_mse / np.var(Y)
    metrics = {'raw_mse': raw_mse, 'normalized_mse': normalized_mse,
               'corr': _cossim(Y - Y.mean(), Y_hat - Y_hat.mean()),
               'cossim': _cossim(Y, Y_hat),  # 'bias': diffs.mean(),
               'y_mean': Y.mean(), 'y_std': Y.std(),
               'yhat_std': Y_hat.std(), 'yhat_mean': Y_hat.mean()}
    problem = task.info['problem']
    metrics['problem'] = problem
    if problem == 'softmax':
        lbls = task.info['lbls_test'].astype(np.int32) # (10000,)
        b = task.info['biases'] # (10,)
        t = time.perf_counter()
        logits_amm = Y_hat + b # (10000, 10) + (10,)
        logits_orig = Y + b
        lbls_amm = np.argmax(logits_amm, axis=1).astype(np.int32) # (10000,)
        lbls_orig = np.argmax(logits_orig, axis=1).astype(np.int32)
        duration_secs = time.perf_counter() - t
        print(f"+bias, softmax time: {duration_secs}")
        # print("Y_hat shape : ", Y_hat.shape)
        # print("lbls hat shape: ", lbls_amm.shape)
        # print("lbls amm : ", lbls_amm[:20])
        metrics['acc_amm'] = np.mean(lbls_amm == lbls)
        metrics['acc_orig'] = np.mean(lbls_orig == lbls)
    
    metrics['secs'] = duration_secs
    
    metrics.update(est.get_speed_metrics(
        task.X_test, task.W_test, fixedB=fixedB))
    
    ##### 3. save downstream metrics to the metric.csv saved from c++ AMM #####
    import csv
    with open(intellistream_amm_metric_save_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Iterate through the dictionary items and write each item as a row
        for key, value in metrics.items():
            row_to_append = [key, value, type(value).__name__]
            writer.writerow(row_to_append)

    return metrics


def _get_all_independent_vars():
    independent_vars = set(['task_id', 'method', 'trial'])
    for method_id in methods.ALL_METHODS:
        hparams = _hparams_for_method(method_id)[0]
        est = _estimator_for_method_id(method_id, **hparams)
        independent_vars = (independent_vars |
                            set(est.get_params().keys()))
    return independent_vars


# @functools.lru_cache(maxsize=None)
# @_memory.cache
def _fitted_est_for_hparams(method_id, hparams_dict, X_train, W_train,
                            Y_train, **kwargs):

    # save X_train, W_train for raw VQ and PQ codebook training
    # print(X_train.shape, W_train.shape)
    # breakpoint()
    # import torch
    # from torch import nn
    # class TensorContainer(nn.Module):
    #     def __init__(self, tensor_dict):
    #         super().__init__()
    #         for key,value in tensor_dict.items():
    #             setattr(self, key, value)
    # tensor_dict = {
    #     "A": torch.from_numpy(X_train),
    #     "B": torch.from_numpy(W_train)
    # }
    # tensors = TensorContainer(tensor_dict)
    # tensors = torch.jit.script(tensors)
    # tensors.save('/home/yuhao/Documents/work/SUTD/AMM/codespace/NEWAMMBENCH/AMMBench/benchmark/datasets/cifar/cifar100_train.pth')
    
    est = _estimator_for_method_id(method_id, **hparams_dict)
    est.fit(X_train, W_train, Y=Y_train, **kwargs)
    return est


# def _main(tasks, methods=['SVD'], saveas=None, ntasks=None,
def _main(tasks_func, methods=None, saveas=None, ntasks=None,
          verbose=1, limit_ntasks=-1, compression_metrics=False, # TODO uncomment below
          # verbose=3, limit_ntasks=-1, compression_metrics=False,
          tasks_all_same_shape=False, intellistream_amm_config_load_path=None, intellistream_amm_metric_save_path=None):
    
    methods = methods.DEFAULT_METHODS if methods is None else methods
    if isinstance(methods, str):
        methods = [methods]
    if limit_ntasks is None or limit_ntasks < 1:
        limit_ntasks = np.inf
    independent_vars = _get_all_independent_vars()

    for method_id in methods:
        if verbose > 0:
            print("running method: ", method_id)
        ntrials = _ntrials_for_method(method_id=method_id, ntasks=ntasks)
        # for hparams_dict in _hparams_for_method(method_id)[2:]: # TODO rm
        for hparams_dict in _hparams_for_method(method_id): # method_id: 'Mithral'
            if verbose > 3:
                print("got hparams: ")
                pprint.pprint(hparams_dict)

            metrics_dicts = []
            try:
                prev_X_shape, prev_Y_shape = None, None
                prev_X_std, prev_Y_std = None, None
                est = None
                for i, task in enumerate(tasks_func()):
                    # task: MatmulTask(X_train, Y_train, X_test, Y_test, W, name='CIFAR-10 Softmax', info=info)
                    if i + 1 > limit_ntasks:
                        raise StopIteration()
                    if verbose > 1:
                        print("-------- running task: {} ({}/{})".format(
                            task.name, i + 1, ntasks))
                        task.validate_shapes()  # fail fast if task is ill-formed

                    can_reuse_est = (
                        (i != 0) and (est is not None)
                        and (prev_X_shape is not None)
                        and (prev_Y_shape is not None)
                        and (prev_X_std is not None)
                        and (prev_Y_std is not None)
                        and (task.X_train.shape == prev_X_shape)
                        and (task.Y_train.shape == prev_Y_shape)
                        and (task.X_train.std() == prev_X_std)
                        and (task.Y_train.std() == prev_Y_std))

                    if not can_reuse_est:
                        try:
                            est = _fitted_est_for_hparams( # <python.vq_amm.MithralMatmul object at 0x7fa3ee517ee0>
                                method_id, hparams_dict,
                                task.X_train, task.W_train, task.Y_train)
                        except amm.InvalidParametersException as e:
                            # hparams don't make sense for task (eg, D < d)
                            if verbose > 2:
                                print(f"hparams apparently invalid: {e}")
                            est = None
                            if tasks_all_same_shape:
                                raise StopIteration()
                            else:
                                continue

                        prev_X_shape = task.X_train.shape
                        prev_Y_shape = task.Y_train.shape
                        prev_X_std = task.X_train.std()
                        prev_Y_std = task.Y_train.std()

                    try:
                        # print(f"task {task.name} matrix hashes:")
                        # pprint.pprint(task.hashes())

                        for trial in range(ntrials):
                            # (Pdb) task
                            # <python.matmul_datasets.MatmulTask object at 0x7f8d2db67e80>
                            # (Pdb) est
                            # <python.vq_amm.MithralMatmul object at 0x7f8d2db67eb0>
                            if isinstance(est, amm_methods.METHOD_TO_ESTIMATOR['IntellistreamAMM']):
                                metrics = _eval_amm_for_intellistream_amm(
                                    task, est, intellistream_amm_config_load_path=intellistream_amm_config_load_path,intellistream_amm_metric_save_path=intellistream_amm_metric_save_path,)
                            else:
                                metrics = _eval_amm(
                                    task, est, compression_metrics=compression_metrics)
                            
                            # (Pdb) task.X_test.shape
                            # (10000, 512)
                            # (Pdb) task.W_test.shape
                            # (512, 10)
                            metrics['N'] = task.X_test.shape[0]
                            metrics['D'] = task.X_test.shape[1]
                            metrics['M'] = task.W_test.shape[1]
                            metrics['trial'] = trial
                            metrics['method'] = method_id
                            metrics['task_id'] = task.name
                            # metrics.update(hparams_dict)
                            metrics.update(est.get_params())
                            print("got metrics: ")
                            pprint.pprint(metrics)
                            # pprint.pprint({k: metrics[k] for k in 'method task_id normalized_mse'.split()})
                            # print("{:.5f}".format(metrics['normalized_mse'])) # TODO uncomment above
                            metrics_dicts.append(metrics)
                    except amm.InvalidParametersException as e:
                        if verbose > 2:
                            print(f"hparams apparently invalid: {e}")
                        if tasks_all_same_shape:
                            raise StopIteration()
                        else:
                            continue

            except StopIteration:  # no more tasks for these hparams
                pass

            if len(metrics_dicts):
                pyn.save_dicts_as_data_frame(
                    metrics_dicts, save_dir='results/amm', name=saveas,
                    dedup_cols=independent_vars)


# def main_ecg(methods=None, saveas='ecg', limit_nhours=1):
#     tasks = md.load_ecg_tasks(limit_nhours=limit_nhours)
#     return _main(tasks=tasks, methods=methods, saveas=saveas, ntasks=139,
#                  # limit_ntasks=10, compression_metrics=False)
#                  limit_ntasks=5, compression_metrics=True)


def main_caltech(methods=methods.USE_METHODS, saveas='caltech',
                 limit_ntasks=-1, limit_ntrain=-1, filt='sobel'):
    # tasks = md.load_caltech_tasks()
    # tasks = md.load_caltech_tasks(limit_ntrain=100e3, limit_ntest=10e3) # TODO rm after debug
    # tasks = md.load_caltech_tasks(limit_ntrain=-1, limit_ntest=10e3) # TODO rm after debug
    # tasks = md.load_caltech_tasks(limit_ntrain=100e3)
    # tasks = md.load_caltech_tasks(limit_ntrain=500e3)
    # tasks = md.load_caltech_tasks(limit_ntrain=1e6)  # does great
    # tasks = md.load_caltech_tasks(limit_ntrain=15e5)
    # tasks = md.load_caltech_tasks(limit_ntrain=17.5e5) # bad
    # tasks = md.load_caltech_tasks(limit_ntrain=2e6)
    # tasks = md.load_caltech_tasks(limit_ntrain=2.5e6)
    # return _main(tasks=tasks, methods=methods, saveas=saveas,
    # limit_ntasks = -1
    # limit_ntasks = 10
    # filt = 'sharpen5x5'
    # filt = 'gauss5x5'
    # filt = 'sobel'
    saveas = '{}_{}'.format(saveas, filt)
    # saveas = '{}_{}'.format(saveas, filt)
    # limit_ntrain = -1
    # limit_ntrain = 500e3
    task_func = functools.partial(
        md.load_caltech_tasks, filt=filt, limit_ntrain=limit_ntrain)
    return _main(tasks_func=task_func, methods=methods,
                 saveas=saveas, ntasks=510, limit_ntasks=limit_ntasks,
                 tasks_all_same_shape=True)


def main_ucr(methods=methods.USE_METHODS, saveas='ucr',
             k=128, limit_ntasks=None, problem='rbf'):
    # limit_ntasks = 10
    # limit_ntasks = 13
    # tasks = md.load_ucr_tasks(limit_ntasks=limit_ntasks)
    # k = 128
    tasks_func = functools.partial(
        md.load_ucr_tasks, limit_ntasks=limit_ntasks, k=k, problem=problem)
    saveas = '{}_k={}_problem={}'.format(saveas, k, problem)
    return _main(tasks_func=tasks_func, methods=methods, saveas=saveas,
                 ntasks=76, limit_ntasks=limit_ntasks,
                 tasks_all_same_shape=False)


def main_cifar10(methods=methods.USE_METHODS, saveas='cifar10'):
    # tasks = md.load_cifar10_tasks()
    return _main(tasks_func=md.load_cifar10_tasks, methods=methods,
                 saveas=saveas, ntasks=1) # md.load_cifar10_tasks: <function load_cifar10_tasks at 0x7f21f4c16680>


def main_cifar100(methods=methods.USE_METHODS, saveas='cifar100'):
    # tasks = md.load_cifar100_tasks()
    return _main(tasks_func=md.load_cifar100_tasks, methods=methods,
                 saveas=saveas, ntasks=1)


def main_all(methods=methods.USE_METHODS):
    main_cifar10(methods=methods)
    main_cifar100(methods=methods)
    # main_ecg(methods=methods)
    main_caltech(methods=methods)


def main():
    # main_cifar10(methods='ScalarQuantize')
    # main_cifar100(methods='ScalarQuantize')
    # main_ucr(methods='ScalarQuantize')
    # main_caltech(methods='ScalarQuantize', filt='sobel')
    # main_caltech(methods='ScalarQuantize', filt='dog5x5')

    # main_cifar10(methods='FastJL')
    # main_cifar10(methods='Mithral')
    # main_cifar10(methods='MithralPQ')
    main_cifar100(methods='Mithral')
    # main_cifar100(methods='MithralPQ')
    # main_ucr(methods='MithralPQ', k=64, limit_ntasks=5, problem='rbf')
    # main_ucr(methods='Bolt', k=64, limit_ntasks=5, problem='softmax')

    # rerun mithral stuff with fixed numerical issues
    # main_cifar10(methods=['Mithral', 'MithralPQ'])
    # main_cifar100(methods=['Mithral', 'MithralPQ'])
    # main_ucr(methods=['Mithral', 'MithralPQ'], k=128, problem='rbf')
    # main_caltech(methods=['Mithral', 'MithralPQ'], filt='sobel')
    # main_caltech(methods=['Mithral', 'MithralPQ'], filt='dog5x5')

    # #
    # # TODO ideally run this too to put in appendix
    # #
    # use_methods = list(methods.USE_METHODS)
    # use_methods.remove(methods.METHOD_SPARSE_PCA)
    # main_ucr(methods=use_methods, k=128, problem='softmax')

    # main_caltech('Mithral', filt='sobel', limit_ntrain=1e6, limit_ntasks=10)
    # lim = 500e3
    # lim = 2e6
    # lim = -1
    # lim = 4e6
    # lim = 5e6
    # main_caltech('Mithral', filt='sobel', limit_ntrain=lim, limit_ntasks=10)
    # main_caltech('MithralPQ', filt='sobel', limit_ntrain=lim, limit_ntasks=10)
    # main_caltech('Mithral', filt='dog5x5', limit_ntrain=lim, limit_ntasks=10)
    # main_caltech('MithralPQ', filt='dog5x5', limit_ntrain=lim, limit_ntasks=10)
    # main_caltech('OldMithralPQ', filt='sobel', limit_ntrain=lim, limit_ntasks=10)

    # main_ucr(methods='MithralPQ', limit_ntasks=5)
    # main_caltech(methods='Bolt', limit_ntasks=10, limit_ntrain=500e3, filt='dog5x5')
    # main_caltech(methods='Bolt', limit_ntasks=10, limit_ntrain=500e3, filt='sobel')
    # main_caltech(methods='SparsePCA')

def main_intellistream_amm(
    task,
    config_load_path,
    metric_save_path
):
    if task=='cifar10':
        _main(tasks_func=md.load_cifar10_tasks, methods='IntellistreamAMM',
                    saveas='tmp', ntasks=1,
                    intellistream_amm_config_load_path=config_load_path,
                    intellistream_amm_metric_save_path=metric_save_path)
    elif task=='cifar100':
        _main(tasks_func=md.load_cifar100_tasks, methods='IntellistreamAMM',
                 saveas='tmp', ntasks=1,
                 intellistream_amm_config_load_path=config_load_path,
                 intellistream_amm_metric_save_path=metric_save_path)
    else:
        raise ValueError("For intellistream amm, we only evaluate on cifar10 and cifar100")

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda f: "{:.2f}".format(f)},
                        linewidth=100)
    
    import argparse

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-t", "--task", default=None, type=str, help="cifar10 or cifar100")
    argParser.add_argument("-c", "--config_load_path", default=None, type=str, help="path to load c++ amm config")
    argParser.add_argument("-m", "--metric_save_path", default=None, type=str, help="path to save c++ amm and downstream metrics")

    args = argParser.parse_args()
    # print(args.config_load_path, args.metric_save_path)

    if args.config_load_path==None and args.metric_save_path==None:
        main()

    elif isinstance(args.config_load_path, str) and isinstance(args.metric_save_path, str):
        main_intellistream_amm(
            task = str(args.task),
            config_load_path = str(args.config_load_path),
            metric_save_path = str(args.metric_save_path)
        )


    # example to call this module:
    # yuhao@yuhao-EUL-WX9  ~/.../bolt/experiments   intellistream_amm ●  python3 -m python.amm_main --config_load_path "/home/yuhao/Documents/work/SUTD/AMM/codespace/coofd_cifar.csv" --metric_save_path "/home/yuhao/Documents/work/SUTD/AMM/codespace/amm_cifar.csv"