import jax
import diffrax
import jax.numpy as jnp 
import functools
from diffrax import diffeqsolve, ODETerm, Dopri5,Tsit5, Dopri8,SaveAt

@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
def forward(model, x, t):
    return model(x, t)

@jax.jit
def fwd_ode(flow_model, x_and_logpx):
  t0 = 0.
  t1 = 1.
  dt0 = t1 - t0
  # flow_model.eval()
  # vector_field = lambda t, x, args: flow_model(x, jnp.full(x.shape[0], t))
  vector_field = lambda t, x, args: forward(flow_model, x, t*jnp.ones((x.shape[0],1)))
  term = ODETerm(vector_field)
  sol = diffeqsolve(term, diffrax.Tsit5(), t0, t1, dt0, x_and_logpx,
                    stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
                    saveat=diffrax.SaveAt(ts=jnp.array([0., 1.])))
  z_and_logpz = sol.ys[-1,:,:]
  z = z_and_logpz[:,:-1]
  log_pz = z_and_logpz[:,-1]
  return z, log_pz

@jax.jit
def rev_ode(flow_model, z_and_logpz):
  t0 = 0.
  t1 = 1.
  dt0 = t1 - t0
  vector_field = lambda t, x, args: forward(flow_model, x, t*jnp.ones((x.shape[0],1)))
  term = ODETerm(vector_field)
  sol = diffeqsolve(term, diffrax.Tsit5(), t1, t0, -dt0, z_and_logpz, stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6), saveat=diffrax.SaveAt(ts=jnp.array([1., 0.])))
  x_and_logpx = sol.ys[-1,:,:]
  x = x_and_logpx[:,:-1]
  log_jac = x_and_logpx[:,-1:]
  log_px = log_jac
  # log_px_true = bimod_dist.log_prob(x)[:,None]
  # log_px_true = jnp.log(bimod_dist.prob(x))[:,None]
  return x, log_px