import jax
import jax.random as jrnd
from typing import Callable
from jaxtyping import Array
from jax import lax,vmap
from jax._src import prng

jax.config.update("jax_enable_x64", True)


# def batch_generator(key: prng.PRNGKeyArray, batch_size: int, prior_dist: Callable):
#     """
#     Generator that yields batches of samples from the prior distribution.

#     Parameters
#     ----------
#     key : prng.PRNGKeyArray
#         Key to generate random numbers.
#     batch_size : int
#         Size of the batch.
#     prior_dist : Callable
#         Prior distribution.

#     """
#     while True:
#         _, key = jrnd.split(key)
#         samples = prior_dist.sample(seed=key, sample_shape=batch_size)
#         logp_samples = prior_dist.log_prob(samples)
#         samples0 = lax.concatenate(
#             (samples, logp_samples[:,None]), 1)

#         _, key = jrnd.split(key)
#         samples = prior_dist.sample(seed=key, sample_shape=batch_size)
#         logp_samples = prior_dist.log_prob(samples)
#         samples1 = lax.concatenate(
#             (samples, logp_samples[:,None]), 1)

#         yield lax.concatenate((samples0, samples1), 0)

def batch_generator(key: prng.PRNGKeyArray, batch_size: int, prior_dist: Callable):
    """
    Generator that yields batches of samples from the prior distribution.

    Parameters
    ----------
    key : prng.PRNGKeyArray
        Key to generate random numbers.
    batch_size : int
        Size of the batch.
    prior_dist : Callable
        Prior distribution.

    """
    v_score = vmap(jax.jacrev(lambda x:
                              prior_dist.log_prob(x)))
    # v_score = jax.vmap(jax.grad(lambda x:
    #                           prior_dist.log_prob(x).sum()))
    while True:
        _, key = jrnd.split(key)
        samples = prior_dist.sample(seed=key, sample_shape=batch_size)
        logp_samples = prior_dist.log_prob(samples)
        score = v_score(samples)
        samples0 = lax.concatenate(
            (samples, logp_samples[:,None], score), 1)

        _, key = jrnd.split(key)
        samples = prior_dist.sample(seed=key, sample_shape=batch_size)
        logp_samples = prior_dist.log_prob(samples)
        score = v_score(samples)
        samples1 = lax.concatenate(
            (samples, logp_samples[:,None], score), 1)
        yield lax.concatenate((samples0, samples1), 0)