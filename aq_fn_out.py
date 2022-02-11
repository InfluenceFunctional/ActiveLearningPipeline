import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


def get_acq_fn(args):
    if args.acq_fn.lower() == "ucb":
        return UCB
    elif args.acq_fn.lower() == "ei":
        return EI
    else:
        return NoAF


class AcquisitionFunctionWrapper():
    def __init__(self, args, model, l2r, dataset):
        self.model = model
        self.l2r = l2r

    def __call__(self, x):
        raise NotImplementedError()

    def update(self, data):
        self.fit(data)

    def fit(self, data):
        self.model.fit(data, reset=True)


class NoAF(AcquisitionFunctionWrapper):
    def __call__(self, x):
        return self.l2r(self.model(x))


class UCB(AcquisitionFunctionWrapper):
    def __init__(self, args, model, l2r, dataset):
        super().__init__(args, model, l2r, dataset)
        self.kappa = args.kappa

    def __call__(self, x):
        mean, std = self.model.forward_with_uncertainty(x)
        return self.l2r(mean + self.kappa * std)


class EI(AcquisitionFunctionWrapper):
    def __init__(self, args, model, l2r, dataset):
        super().__init__(args, model, l2r, dataset)
        self.args = args
        self.device = args.device
        self.sigmoid = nn.Sigmoid()
        self.best_f = self._get_best_f(dataset)

    def _get_best_f(self, dataset):
        f_values = []
        data_it = dataset.pos_train if self.args.proxy_type == "classification" else dataset.train
        for sample in data_it:
            outputs = self.model([sample])
            if self.args.proxy_type == "classification":
                outputs = self.sigmoid(outputs)
            f_values.append(outputs.item())
        return torch.tensor(np.percentile(f_values, self.args.max_percentile))

    def __call__(self, x):
        self.best_f = self.best_f.to(self.device)
        outputs = [self.model(x, True) for _ in range(self.args.proxy_num_dropout_samples)]
        outputs = torch.cat([self.l2r(outputs[i]) for i in range(len(outputs))])
        mean, sigma = outputs.mean(dim=0), outputs.std(dim=0)
        # deal with batch evaluation and broadcasting
        # mean = mean.view(view_shape)
        # sigma = sigma.view(view_shape)

        u = (mean - self.best_f.expand_as(mean)) / sigma

        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        return ei.cpu().numpy()

    def update(self, data):
        self.fit(data, reset=True)
        self.best_f = self._get_best_f(data)