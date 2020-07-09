#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import opt_einsum as oe
import torch
from torch.functional import F

from .utils import get_layer_type, sum_over_all_but_batch_and_last_n


def _compute_linear_grad_sample(layer, A, B):
    # gs = oe.contract("n...i,n...j->n...ij", B, A)
    # layer.weight.grad_sample = oe.contract("n...ij->nij", gs)
    A = A.reshape(A.size()[0], -1, A.size()[-1])
    B = B.reshape(B.size()[0], -1, B.size()[-1])
    layer.weight.grad_sample_norm = torch.sqrt(
        oe.contract('nxi,nxj,nyi,nyj->n', B, A, B, A, optimize='auto'))
    layer.weight.grad_sample_weighted_sum = (
        lambda w: oe.contract('nxi,nxj,n->ij', B, A, w, optimize='auto'))

    if layer.bias is not None:
        layer.bias.grad_sample_norm = torch.sqrt(
            oe.contract('nxi,nyi->n', B, B, optimize='auto'))
        layer.bias.grad_sample_weighted_sum = (
            lambda w: oe.contract('nxi,n->i', B, w, optimize='auto'))


def _compute_sequence_bias_grad_sample(layer, A, B):
    raise NotImplementedError()
    _create_or_extend_grad_sample(layer.bias, B[:, -1])


def _compute_norm_grad_sample(layer, A, B):
    raise NotImplementedError()
    layer_type = get_layer_type(layer)
    if layer_type == "LayerNorm":
        _create_or_extend_grad_sample(
            layer.weight,
            sum_over_all_but_batch_and_last_n(
                F.layer_norm(A, layer.normalized_shape, eps=layer.eps) * B,
                layer.weight.dim(),
            ),
            batch_dim,
        )
        _create_or_extend_grad_sample(
            layer.bias,
            sum_over_all_but_batch_and_last_n(B, layer.bias.dim()),
            batch_dim,
        )
    elif layer_type == "GroupNorm":
        gs = F.group_norm(A, layer.num_groups, eps=layer.eps) * B
        _create_or_extend_grad_sample(
            layer.weight, oe.contract("ni...->ni", gs)
        )
        if layer.bias is not None:
            _create_or_extend_grad_sample(
                layer.bias, oe.contract("ni...->ni", B)
            )
    elif layer_type in {"InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d"}:
        gs = F.instance_norm(A, eps=layer.eps) * B
        _create_or_extend_grad_sample(
            layer.weight, oe.contract("ni...->ni", gs)
        )
        if layer.bias is not None:
            _create_or_extend_grad_sample(
                layer.bias, oe.contract("ni...->ni", B)
            )


def _compute_conv_grad_sample(layer, A, B):
    n = A.shape[0]
    layer_type = get_layer_type(layer)
    # get A and B in shape depending on the Conv layer
    if layer_type == "Conv2d":
        A = torch.nn.functional.unfold(A, layer.kernel_size, padding=layer.padding, stride=layer.stride)
        B = B.reshape(n, -1, A.shape[-1])
    elif layer_type == "Conv1d":
        raise NotImplementedError()
        # unfold doesn't work for 3D tensors; so force it to be 4D
        A = A.unsqueeze(-2)  # add the H dimension
        # set arguments to tuples with appropriate second element
        A = torch.nn.functional.unfold(
            A,
            (1, layer.kernel_size[0]),
            padding=(0, layer.padding[0]),
            stride=(1, layer.stride[0]),
        )
        B = B.reshape(n, -1, A.shape[-1])
    try:
        # n=batch_sz; o=num_out_channels; p=num_in_channels*kernel_sz
        layer.weight.grad_sample_norm = torch.sqrt(
            oe.contract('noq,npq,nor,npr->n', B, A, B, A, optimize='auto')
            if layer.groups == 1
            else oe.contract("njk,njk,njl,njl->n", B, A, B, A, optimize='auto'))
        layer.weight.grad_sample_weighted_sum = (
            (lambda w: oe.contract("noq,npq,n->op", B, A, w, optimize='auto')
                            .reshape(layer.weight.shape))
            if layer.groups == 1
            else (lambda w: oe.contract("njk,njk->j", B, A, w, optimize='auto')
                                 .reshape(layer.weight.shape))
        )
    except Exception as e:
        raise type(e)(
            f"{e} There is probably a problem with {layer_type}.groups"
            + "It should be either 1 or in_channel"
        )
    if layer.bias is not None:
        layer.bias.grad_sample_norm = torch.sqrt(
            oe.contract('noq,npq->n', B, B, optimize='auto'))
        layer.bias.grad_sample_weighted_sum = (
            lambda w: oe.contract("noq,n->o", B, w, optimize='auto'))


def _compute_embedding_grad_sample(layer, A, B):
    raise NotImplementedError()
    one_hot = F.one_hot(A, num_classes=layer.weight.shape[0])
    gs = oe.contract("n...i,n...j->n...ij", one_hot, B)

    _create_or_extend_grad_sample(
        layer.weight, oe.contract("n...ij->nij", gs)
    )


_supported_layers_grad_samplers = {
    "Embedding": _compute_embedding_grad_sample,
    "Linear": _compute_linear_grad_sample,
    "Conv2d": _compute_conv_grad_sample,
    "Conv1d": _compute_conv_grad_sample,
    "LayerNorm": _compute_norm_grad_sample,
    "GroupNorm": _compute_norm_grad_sample,
    "InstanceNorm1d": _compute_norm_grad_sample,
    "InstanceNorm2d": _compute_norm_grad_sample,
    "InstanceNorm3d": _compute_norm_grad_sample,
    "SequenceBias": _compute_sequence_bias_grad_sample,
}  # Supported layer class types
