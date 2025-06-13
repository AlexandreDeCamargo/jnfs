import jax 
import jax.numpy as jnp
import equinox as eqx


class Flow(eqx.Module):
    linear_in: eqx.nn.Linear
    blocks: list[eqx.nn.Linear]
    linear_out: eqx.nn.Linear

    def __init__(self, din: int, dim: int, key: jax.random.PRNGKey):
        self.linear_in = eqx.nn.Linear(din + 1, dim, key=key)
        self.blocks = [eqx.nn.Linear(dim, dim, key=subkey) for subkey in jax.random.split(key, 3)]
        self.linear_out = eqx.nn.Linear(dim, din, key=key)

    def __call__(self, x, t):
        x = jnp.concatenate([x, t], axis=-1)
        x = self.linear_in(x)
        x = jnp.tanh(x)
        for block in self.blocks:
            x = block(x)
            x = jnp.tanh(x)
        x = self.linear_out(x)
        return x

class CNF(eqx.Module):
    flow: Flow

    def __init__(self, din: int, dim: int, key: jax.random.PRNGKey):
        self.flow = Flow(din, dim, key=key)

    def __call__(self, states, t):
        x, log_px = states[:-1], states[-1:]
        dz, f_vjp = jax.vjp(self.flow, x, t)
        x_ones = jnp.ones((self.flow.linear_out.out_features,))
        (dtrJ, _) = f_vjp(x_ones)
        dtrJ = jnp.sum(dtrJ)
        return jnp.concatenate([dz, -dtrJ[None]], axis=-1)