import jax
import diffrax
import jax.numpy as jnp 
import functools
from diffrax import diffeqsolve, ODETerm,Tsit5 
from jaxtyping import Array, Float, PyTree

@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
def forward(model, x, t):
    return model(x, t)


def fwd_ode(flow_model: PyTree, x_and_logpx, data_dim: int):
  t0 = 0.
  t1 = 1.
  dt0 = t1 - t0
  vector_field = lambda t, x, args: forward(flow_model, x, t*jnp.ones((x.shape[0],1)))
  term = ODETerm(vector_field)
  sol = diffeqsolve(term, 
                    Tsit5(), 
                    t0, 
                    t1, 
                    dt0, 
                    x_and_logpx,
                    stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
                    saveat=diffrax.SaveAt(ts=jnp.array([0., 1.])))
  z_t1= sol.ys[-data_dim, :, 0:data_dim]           # Shape (batch, 1)
  logp_diff_t1 = sol.ys[-data_dim, :, data_dim:data_dim+1]   # Shape (batch, 1)
  score_t1 = sol.ys[-data_dim, :, data_dim+1:data_dim+2]       # Shape (batch, 1)
  return z_t1, logp_diff_t1, score_t1

def rev_ode(flow_model: PyTree, z_and_logpz, data_dim: int):
  t0 = 0.
  t1 = 1.
  dt0 = t1 - t0
  vector_field = lambda t, x, args: forward(flow_model, x, t*jnp.ones((x.shape[0],1)))
  term = ODETerm(vector_field)
  sol = diffeqsolve(term, 
                    Tsit5(), 
                    t1, 
                    t0,
                    -dt0,
                    z_and_logpz,
                    stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6), 
                    saveat=diffrax.SaveAt(ts=jnp.array([1., 0.])))
  x_and_logpx = sol.ys[-data_dim,:,:]
  x = x_and_logpx[:,:-data_dim]
  log_jac = x_and_logpx[:,-data_dim:]
  log_px = log_jac
  return x, log_px