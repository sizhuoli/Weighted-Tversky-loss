#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

#weighted_tversky_loss
def tversky(y_true, y_pred, alpha=0.6, beta=0.4):
    """compute the weighted Tversky loss with weight maps"""
    #annotation
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    #weights
    y_weights = y_true[...,1]
    y_weights = y_weights[...,np.newaxis]
    ones = K.ones(K.shape(y_t))
    #p0: prob that the pixel is of class 1
    p0 = y_pred  
    #p1: prob that the pixel is of class 0
    p1 = ones - y_pred  
    g0 = y_t
    g1 = ones - y_t
    #terms in the Tversky loss function combined with weights
    tp = tf.reduce_sum(y_weights * p0 * g0)
    fp = alpha * tf.reduce_sum(y_weights * p0 * g1)
    fn = beta * tf.reduce_sum(y_weights * p1 * g0)
    #add to the denominator a small epsilon to prevent the value from being undefined 
    EPS = 1e-5
    num = tp
    den = tp + fp + fn + EPS
    result = num / den
    return 1 - tf.reduce_mean(result)