import jax 
import jax.numpy as jnp
import equinox as eqx
from typing import Any 

@jax.jit
def thomas_fermi_1D(den: Any, Ne: int, c: float=(jnp.pi*jnp.pi)/24) -> jax.Array:
    r"""
    Thomas-Fermi kinetic functional in 1D.
    See original paper eq. 18 in https://pubs.aip.org/aip/jcp/article/139/22/224104/193579/Orbital-free-bond-breaking-via-machine-learning

    T_{\text{TF}}[\rhom] = \frac{\pi^2}{24} \int \left(\rhom(x) \right)^{3} \mathrm{d}x \\
    T_{\text{TF}}[\rhom] = \frac{\pi^2}{24} \Ne^3 \EX_{\rhozero} \left[ (\rhophi(x))^{2}

    Parameters
    ----------
    den : Array
        Density.
    score : Array
        Gradient of the log-likelihood function.
    Ne : int
        Number of electrons.
    c : float, optional
        Multiplication constant, by default (jnp.pi*jnp.pi)/24

    Returns
    -------
    jax.Array
        Thomas-Fermi kinetic energy.
    """

    den_sqr = den*den
    return c*(Ne**3)*den_sqr

@jax.jit
def soft_coulomb(x:Any,xp:Any,Ne: int) -> jax.Array:
    r"""
    Soft-Coulomb potential.

    See eq 6 in https://pubs.aip.org/aip/jcp/article/139/22/224104/193579/Orbital-free-bond-breaking-via-machine-learning

    Parameters
    ----------
    x : Any
        A point where the potential is evaluated.
    xp : Any
        A point where the charge density is zero.
    Ne : int
        Number of electrons.

    Returns
    -------
    jax.Array
        Soft version of the Coulomb potential.
    """
    v_coul = 1/(jnp.sqrt( 1 + (x-xp)*(x-xp)))
    return v_coul*Ne**2

def attraction(x:Any, R:float, Z_alpha:int, Z_beta:int, Ne: int) -> jax.Array:
    """
    Attraction between two nuclei.

    See eq 7 in https://pubs.aip.org/aip/jcp/article/139/22/224104/193579/Orbital-free-bond-breaking-via-machine-learning

    Parameters
    ----------
    x : Any
        A point where the potential is evaluated.
    R : float
        Distance between the two nuclei.
    Z_alpha : int
        Atomic number of the first nucleus.
    Z_beta : int
        Atomic number of the second nucleus.
    Ne : int
        Number of electrons.

    Returns
    -------
    jax.Array
        Attraction to the nuclei of charges Z_alpha and Z_beta.
    """
    v_x = - Z_alpha/(jnp.sqrt(1 + (x + R/2)**2))  - Z_beta/(jnp.sqrt(1 + (x - R/2)**2))
    return v_x*Ne

@jax.jit
def exchange_correlation_one_dimensional(den:Any, Ne:int) -> jax.Array:
    """
    1D exchange-correlation functional
    See eq 7 in https://iopscience.iop.org/article/10.1088/1751-8113/42/21/214021

    \epsilon_{\text{XC}} (\rs,\zeta) = \frac{\azeta + \bzeta \rs + \czeta \rs^{2}}{1 + \dzeta \rs + \ezeta \rs^2 + \fzeta \rs^3} + \frac{\gzeta \rs \ln[{\rs +
                                        + \alphazeta \rs^{\betazeta} }]}{1 + \hzeta \rs^2}

    Parameters
    ----------
    den : Array
        Density.
    Ne : int
        Number of electrons.

    Returns
    -------
    jax.Array
        Exchange-correlation energy.
    """
    rs = 1/(2*Ne*den)
    a0 = -0.8862269
    b0 = -2.1414101
    c0 = 0.4721355
    d0 = 2.81423
    e0 = 0.529891
    f0 = 0.458513
    g0 = -0.202642
    h0 = 0.470876
    alpha0 = 0.104435
    beta0 = 4.11613
    n1 = a0 + b0*rs + c0*rs**2
    d1 = 1 + d0*rs + e0*rs**2 + f0*rs**3
    f1 = n1/d1
    n2 = g0*rs*jnp.log(rs + alpha0*rs**beta0)
    d2 = 1 + h0*rs**2
    f2 = n2/d2
    return Ne*(f1 + f2)