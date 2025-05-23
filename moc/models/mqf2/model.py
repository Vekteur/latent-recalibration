# Adapted from https://github.com/awslabs/gluonts/tree/dev/src/gluonts/torch/model/mqf2
# Copyright (c) 2021 Chin-Wei Huang

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from cpflows.flows import DeepConvexFlow, SequentialFlow
from cpflows.icnn import PosLinear, Softplus, softplus, symm_softplus
from torch import nn


def get_flow(
    input_dim: int,
    output_dim: int,
    icnn_hidden_size: int = 30,
    icnn_num_layers: int = 2,
    is_energy_score: bool = False,
    estimate_logdet: bool = False,
):
    picnn = PICNN(
        dim=output_dim,
        dimh=icnn_hidden_size,
        dimc=input_dim,
        num_hidden_layers=icnn_num_layers,
        symm_act_first=True,
    )
    deepconvexflow = MyDeepConvexFlow(
        picnn,
        output_dim,
        estimate_logdet=estimate_logdet,
    )

    if is_energy_score:
        networks = [deepconvexflow]
    else:
        networks = [
            ActNorm(output_dim),
            deepconvexflow,
            ActNorm(output_dim),
        ]

    return MySequentialFlow(networks, output_dim)


class MyDeepConvexFlow(DeepConvexFlow):
    def __init__(
        self,
        picnn: torch.nn.Module,
        dim: int,
        estimate_logdet: bool = False,
        m1: int = 10,
        m2: Optional[int] = None,
        rtol: float = 0.0,
        atol: float = 1e-3,
    ) -> None:
        super().__init__(
            picnn,
            dim,
            m1=m1,
            m2=m2,
            rtol=rtol,
            atol=atol,
        )

        self.estimate_logdet = estimate_logdet

    def get_potential(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = x.size(0)
        output = self.icnn(x, context)

        return (
            F.softplus(self.w1) * output + F.softplus(self.w0) * (x.view(n, -1) ** 2).sum(1, keepdim=True) / 2
        )

    def forward_transform(
        self,
        x: torch.Tensor,
        logdet: Optional[Union[float, torch.Tensor]] = 0.0,
        context: Optional[torch.Tensor] = None,
        extra: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.estimate_logdet:
            return self.forward_transform_stochastic(x, logdet, context=context, extra=extra)
        return self.forward_transform_bruteforce(x, logdet, context=context)


class MySequentialFlow(SequentialFlow):
    def __init__(self, flows: list[torch.nn.Module], dim: int) -> None:
        super().__init__(flows)
        self.dim = dim

    def forward(self, y: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = context
        batch_shape = x.shape[:-1]
        sample_shape = y.shape[: -(len(batch_shape) + 1)]
        x_repeat = x.view((1,) * len(sample_shape) + x.shape).expand(sample_shape + x.shape)
        y_flat, x_repeat_flat = (
            y.reshape(-1, y.shape[-1]),
            x_repeat.reshape(-1, x_repeat.shape[-1]),
        )

        for flow in self.flows:
            if isinstance(flow, MyDeepConvexFlow):
                y_flat = flow.forward(y_flat, context=x_repeat_flat)
            elif isinstance(flow, ActNorm):
                y_flat = flow.forward_transform(y_flat)[0]

        return y_flat.reshape(sample_shape + batch_shape + (self.dim,))

    def set_estimate_log_det(self, estimate_logdet: bool):
        for flow in self.flows:
            if isinstance(flow, MyDeepConvexFlow):
                flow.estimate_logdet = estimate_logdet


# We modify ActNorm from https://github.com/CW-Huang/CP-Flow/blob/main/lib/flows/flows.py such that the attribute
# `initialized` is registered as a buffer. This is necessary to ensure that the initialization status is persisted
# after loading a checkpoint.

_scaling_min = 0.001


# noinspection PyUnusedLocal
class ActNorm(torch.nn.Module):
    """ActNorm layer with data-dependant init."""

    def __init__(self, num_features, logscale_factor=1.0, scale=1.0, learn_scale=True):
        super().__init__()
        self.register_buffer('initialized', torch.tensor(False))  # Register as buffer
        self.num_features = num_features

        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True))
        self.learn_scale = learn_scale
        if learn_scale:
            self.logscale_factor = logscale_factor
            self.scale = scale
            self.register_parameter(
                'logs',
                nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True),
            )

    def forward_transform(self, x, logdet=0):
        input_shape = x.size()
        x = x.view(input_shape[0], input_shape[1], -1)

        if not self.initialized.item():  # Convert buffer tensor to Python boolean
            self.initialized.fill_(True)  # Persist initialization status

            # noinspection PyShadowingNames
            def unsqueeze(x):
                return x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = x.size(0) * x.size(-1)
            b = -torch.sum(x, dim=(0, -1)) / sum_size
            self.b.data.copy_(unsqueeze(b).data)

            if self.learn_scale:
                var = unsqueeze(torch.sum((x + unsqueeze(b)) ** 2, dim=(0, -1)) / sum_size)
                logs = torch.log(self.scale / (torch.sqrt(var) + 1e-6)) / self.logscale_factor
                self.logs.data.copy_(logs.data)

        b = self.b
        output = x + b

        if self.learn_scale:
            logs = self.logs * self.logscale_factor
            scale = torch.exp(logs) + _scaling_min
            output = output * scale
            dlogdet = torch.sum(torch.log(scale)) * x.size(-1)  # c x h

            return output.view(input_shape), logdet + dlogdet
        return output.view(input_shape), logdet

    def reverse(self, y, **kwargs):
        assert self.initialized.item()  # Ensure initialization happened
        input_shape = y.size()
        y = y.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        scale = torch.exp(logs) + _scaling_min
        x = y / scale - b

        return x.view(input_shape)

    def extra_repr(self):
        return f'{self.num_features}'


class ActNormNoLogdet(ActNorm):
    def forward(self, x):
        return super().forward_transform(x)[0]


# We modify PICNN from https://github.com/CW-Huang/CP-Flow/blob/main/lib/flows/icnn.py to use the ActNorm objects
# defined above.


# noinspection PyPep8Naming,PyTypeChecker
class PICNN(torch.nn.Module):
    def __init__(
        self,
        dim=2,
        dimh=16,
        dimc=2,
        num_hidden_layers=2,
        PosLin=PosLinear,
        symm_act_first=False,
        softplus_type='gaussian_softplus',
        zero_softplus=False,
    ):
        super().__init__()
        # with data dependent init

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.act_c = nn.ELU()
        self.symm_act_first = symm_act_first

        # data path
        Wzs = []
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLin(dimh, dimh, bias=True))
        Wzs.append(PosLin(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        # skip data
        Wxs = []
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        # context path
        Wcs = []
        Wcs.append(nn.Linear(dimc, dimh))
        self.Wcs = torch.nn.ModuleList(Wcs)

        Wczs = []
        for _ in range(num_hidden_layers - 1):
            Wczs.append(nn.Linear(dimh, dimh))
        Wczs.append(nn.Linear(dimh, dimh, bias=True))
        self.Wczs = torch.nn.ModuleList(Wczs)
        for Wcz in self.Wczs:
            Wcz.weight.data.zero_()
            Wcz.bias.data.zero_()

        Wcxs = []
        for _ in range(num_hidden_layers - 1):
            Wcxs.append(nn.Linear(dimh, dim))
        Wcxs.append(nn.Linear(dimh, dim, bias=True))
        self.Wcxs = torch.nn.ModuleList(Wcxs)
        for Wcx in self.Wcxs:
            Wcx.weight.data.zero_()
            Wcx.bias.data.zero_()

        Wccs = []
        for _ in range(num_hidden_layers - 1):
            Wccs.append(nn.Linear(dimh, dimh))
        self.Wccs = torch.nn.ModuleList(Wccs)

        self.actnorm0 = ActNormNoLogdet(dimh)
        actnorms = []
        for _ in range(num_hidden_layers - 1):
            actnorms.append(ActNormNoLogdet(dimh))
        actnorms.append(ActNormNoLogdet(1))
        self.actnorms = torch.nn.ModuleList(actnorms)

        self.actnormc = ActNormNoLogdet(dimh)

    def forward(self, x, c):
        if self.symm_act_first:
            z = symm_softplus(self.actnorm0(self.Wzs[0](x)), self.act)
        else:
            z = self.act(self.actnorm0(self.Wzs[0](x)))
        c = self.act_c(self.actnormc(self.Wcs[0](c)))
        for Wz, Wx, Wcz, Wcx, Wcc, actnorm in zip(
            self.Wzs[1:-1],
            self.Wxs[:-1],
            self.Wczs[:-1],
            self.Wcxs[:-1],
            self.Wccs,
            self.actnorms[:-1],
        ):
            cz = softplus(Wcz(c) + np.exp(np.log(1.0) - 1))
            cx = Wcx(c) + 1.0
            z = self.act(actnorm(Wz(z * cz) + Wx(x * cx) + Wcc(c)))

        cz = softplus(self.Wczs[-1](c) + np.log(np.exp(1.0) - 1))
        cx = self.Wcxs[-1](c) + 1.0
        return self.actnorms[-1](self.Wzs[-1](z * cz) + self.Wxs[-1](x * cx))
