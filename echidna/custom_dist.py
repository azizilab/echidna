import torch
from pyro import distributions as dist

class TruncatedNormal(dist.Distribution):

    '''
    Truncated Normal distribution implemented for Pyro. Derived from numpyro's implementation.
    '''

    def __init__(self, loc, scale, lower=None, upper=None, batch_shape=()):
        super().__init__()
        self.lower = lower if lower is not None else -torch.inf
        self.upper = upper if upper is not None else torch.inf

        loc = loc if isinstance(loc, torch.Tensor) else torch.tensor(loc)
        scale = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale)

        self.batch_shape = torch.broadcast_shapes(
            loc.shape, scale.shape,
        )

        self.base_dist = dist.Normal(loc, scale).expand(self.batch_shape)
        self.event_shape = self.base_dist.event_shape
        self.event_dim = self.base_dist.event_dim

        if self.lower == -torch.inf and self.upper != torch.inf:
            self.support = dist.constraints.less_than(upper)
        elif self.lower != -torch.inf and self.upper == torch.inf:
            self.support = dist.constraints.greater_than(lower)
        else:
            self.support = dist.constraints.interval(lower, upper)

    def expand(self, batch_shape: torch.Size, _instance=None):
        return self.__class__(self.base_dist.loc.expand(batch_shape), self.base_dist.scale.expand(batch_shape), self.lower, self.upper)


    def log_prob(self, value):
        if self.lower == -torch.inf and self.upper != torch.inf:
            # right bound -- (-inf, b]
            return self.base_dist.log_prob(value) - torch.log(self._cdf_at_high_right())
        if self.lower != -torch.inf and self.upper == torch.inf:
            # left bound -- [a, inf)
            sign = torch.where(self.base_dist.loc >= self.lower, 1.0, -1.0)
            return self.base_dist.log_prob(value) - torch.log(
                sign * (self._tail_prob_at_high_left() - self._tail_prob_at_low_left())
                )
        else:
            # two-sided bounds -- [a, b]
            return self.base_dist.log_prob(value) - self._log_diff_tail_probs_both()

    def _tail_prob_at_low_left(self):
        # if low < loc, returns cdf(low); otherwise returns 1 - cdf(low)
        loc = self.base_dist.loc
        sign = torch.where(loc >= self.lower, 1.0, -1.0)
        return self.base_dist.cdf(loc - sign * (loc - self.lower))

    def _tail_prob_at_high_left(self):
        # if low < loc, returns cdf(high) = 1; otherwise returns 1 - cdf(high) = 0
        return torch.where(self.lower <= self.base_dist.loc, 1.0, 0.0)

    def _cdf_at_high_right(self):
        return self.base_dist.cdf(torch.tensor(self.upper))

    def _tail_prob_at_low_both(self):
        # if low < loc, returns cdf(low); otherwise returns 1 - cdf(low)
        loc = self.base_dist.loc
        sign = torch.where(loc >= self.lower, 1.0, -1.0)
        return self.base_dist.cdf(loc - sign * (loc - self.lower))

    def _tail_prob_at_high_both(self):
        # if low < loc, returns cdf(high); otherwise returns 1 - cdf(high)
        loc = self.base_dist.loc
        sign = torch.where(loc >= self.lower, 1.0, -1.0)
        return self.base_dist.cdf(loc - sign * (loc - self.upper))

    def _log_diff_tail_probs_both(self):
        # use log_cdf method, if available, to avoid inf's in log_prob
        # fall back to cdf, if log_cdf not available
        log_cdf = getattr(self.base_dist, "log_cdf", None)
        if callable(log_cdf):
            return torch.logsumexp(
                a=torch.stack([log_cdf(self.upper), log_cdf(self.lower)], axis=-1),
                axis=-1,
                b=torch.Tensor([1, -1]),  # subtract low from high
            )

        else:
            loc = self.base_dist.loc
            sign = torch.where(loc >= self.lower, 1.0, -1.0)
            return torch.log(sign * (self._tail_prob_at_high_both() - self._tail_prob_at_low_both()))

    def _clamp_probs(self, probs):
        finfo = torch.finfo(torch.result_type(probs, 1.))
        return torch.clip(probs, min=finfo.tiny, max=1.0 - finfo.eps)

    def sample(self, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        with torch.no_grad():
            u = torch.rand(shape)
            loc = self.base_dist.loc
            sign = torch.where(loc >= self.lower, 1.0, -1.0)

            if self.lower == -torch.inf and self.upper != torch.inf:
                return self.base_dist.icdf(u * self._cdf_at_high_right())
            if self.lower != -torch.inf and self.upper == torch.inf:
                return (1 - sign) * loc + sign * self.base_dist.icdf(
                    (1 - u) * self._tail_prob_at_low_left() + u * self._tail_prob_at_high_left()
                )
            else:
                return (1 - sign) * loc + sign * self.base_dist.icdf(
                    self._clamp_probs((1 - u) * self._tail_prob_at_low_both() + u * self._tail_prob_at_high_both())
                )