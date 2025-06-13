import distrax
import jax
import jax.numpy as jnp
import jax.random as jrnd
import optax
# from flax import nnx
import equinox as eqx
from typing import Any,List,Tuple,Callable
from optax import ema
import diffrax
from diffrax import ODETerm, Euler, Dopri5, AbstractSolver, diffeqsolve

from diffrax import diffeqsolve, ODETerm, Dopri5,Tsit5, Dopri8,SaveAt
from diffrax import ODETerm, Tsit5, diffeqsolve, PIDController, SaveAt

from tensorflow_probability.substrates import jax as tfp
from jax import lax
import matplotlib
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
import functools
from functools import partial
import chex

from distrax import MultivariateNormalDiag
from flow import Flow,CNF
from ode import fwd_ode
from functionals import *
from utils import batch_generator

base_dist0 = distrax.MultivariateNormalDiag(jnp.array([-2.]), jnp.array([0.5]))
base_dist1 = distrax.MultivariateNormalDiag(jnp.array([2.]), jnp.array([0.3]))
bimod_dist = distrax.MixtureOfTwo(0.5,base_dist0,base_dist1)

data_dim: int = 1
model_dim: int = 264
Ne:int = 2 
R: int = 10 
Z_alpha: int = 3
Z_beta: int = 1
png = jrnd.PRNGKey(0)
_, key = jrnd.split(png)


flow_model = CNF(data_dim, model_dim, key) 
# to accumulate the value fo all functionals
@chex.dataclass
class F_values:
    energy: chex.ArrayDevice
    kin: chex.ArrayDevice
    vnuc: chex.ArrayDevice
    hart: chex.ArrayDevice
    xc: chex.ArrayDevice
    
energies_ema = ema(decay=0.99)
energies_state = energies_ema.init(
    F_values(energy=jnp.array(0.), kin=jnp.array(0.), vnuc=jnp.array(0.),  hart = jnp.array(0.), xc = jnp.array(0.)))

def sample_bimod(key, bs):
  x = bimod_dist.sample(seed=key, sample_shape=(bs,))
  p0 = bimod_dist.prob(x)
  return jnp.concatenate([x,jnp.log(p0)[:,None]], axis=-1)

base_dist = distrax.MultivariateNormalDiag(jnp.array([0.]), jnp.array([1.]))
def sample_pz(key, bs):
  z = base_dist.sample(seed=key, sample_shape=(bs,))
  log_pz = base_dist.log_prob(z)
  return jnp.concatenate([z,log_pz[:,None]], axis=-1)

def grad_loss(model, z_and_logpz):
  x, log_px = fwd_ode(model, z_and_logpz)
  den_all, x_all = jnp.exp(log_px), x
  den, denp = den_all[:-1], den_all[-1:]
  x, xp = x_all[:-1], x_all[-1:]

  # evaluate all the functionals locally F[x_i, \rho(x_i)]
  e_t = thomas_fermi_1D(den, Ne)
  e_h = soft_coulomb(x, xp, Ne)
  e_nuc_v = attraction(x, R, Z_alpha, Z_beta,Ne)
  e_xc = exchange_correlation_one_dimensional(den, Ne)
  e = e_t + e_nuc_v + e_h + e_xc

  energy = jnp.mean(e)

  f_values = F_values(energy=energy,
                            kin=jnp.mean(e_t),
                            vnuc=jnp.mean(e_nuc_v),
                            hart=jnp.mean(e_h),
                            xc = jnp.mean(e_xc)
                            )

  return energy, f_values


@eqx.filter_jit
def train_step(flow_model, optimizer_state, x_and_logpx):
    # Compute loss and gradients
    loss, grads = eqx.filter_value_and_grad(grad_loss, has_aux=True)(flow_model, x_and_logpx)

    # Update the model parameters
    updates, optimizer_state = optimizer.update(grads, optimizer_state,flow_model)
    flow_model = eqx.apply_updates(flow_model, updates)

    return loss, flow_model, optimizer_state

# Define the optimizer
lr = optax.exponential_decay(2e-3, transition_steps=1, decay_rate=0.95)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(lr, weight_decay=1e-5)
)
optimizer_state = optimizer.init(eqx.filter(flow_model, eqx.is_array))

prior_dist = MultivariateNormalDiag(jnp.zeros(1), 1.*jnp.ones(1))

gen_batches = batch_generator(key, model_dim, prior_dist)

# Training loop
for itr in range(10):
    key, subkey = jax.random.split(key)
    x_and_logpx = sample_bimod(subkey, 32)
    x_and_logpx = x_and_logpx.at[:, -1].set(0.0)

    _,key = jrnd.split(key)
    batch = next(gen_batches)

    # Perform a training step
    loss, flow_model, optimizer_state = train_step(flow_model, optimizer_state, batch)
    loss_epoch, losses = loss
    energies_i_ema, energies_state = energies_ema.update(
            losses, energies_state)
    ei_ema = energies_i_ema.energy

    r_ema = {'epoch': itr,
                 'E': energies_i_ema.energy,
                 'T': energies_i_ema.kin, 'V': energies_i_ema.vnuc, 'H': energies_i_ema.hart

                 }
    print( r_ema)

    