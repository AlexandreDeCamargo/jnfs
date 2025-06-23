import jax 
import jax.numpy as jnp
import equinox as eqx
from jax import lax
from typing import Any
from jaxtyping import Array, Float, PyTree

@jax.jit
def weizsacker(den: Float, score: Float, Ne: int, lambda_0: Float=0.2) -> Float[Array, "batch dim"]:
    r"""
    von Weizsacker gradient correction.
    See paper eq. 3 in https://pubs.aip.org/aip/jcp/article/114/2/631/184186/Thomas-Fermi-Dirac-von-Weizsacker-models-in-finite

    T_{\text{Weizsacker}}[\rho] = \frac{\lambda}{8} \int \frac{(\nabla \rho)^2}{\rho} d\boldsymbol{x} =
                                = \frac{\lambda}{8} \int  \rho \left(\frac{(\nabla \rho)}{\rho}\right)^2 d\boldsymbol{x}\\
    T_{\text{Weizsacker}}[\rho] = \mathbb{E}_{\rho} \left[ \left(\frac{(\nabla \rho)}{\rho}\right)^2 \right]

    Parameters
    ----------
    den : Array
        Density.
    score : Array
        Gradient of the log-likelihood function.
    Ne : int
        Number of electrons.
    lambda_0 : float, optional (W Stich, EKU Gross., Physik A Atoms and Nuclei, 309(1):511, 1982.)
        Phenomenological parameter, by default .2

    Returns
    -------
    jax.Array
        Thomas-Weizsacker kinetic energy.
    """
    score_sqr = jnp.einsum('ij,ij->i', score, score)
    return (lambda_0*Ne/8.)*lax.expand_dims(score_sqr, (1,))

@jax.jit
def thomas_fermi_1D(den: Float, Ne: int, c: Float=(jnp.pi*jnp.pi)/24) -> Float[Array, "batch dim"]:
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
def soft_coulomb(x:Float,xp:Float,Ne: int) -> Float[Array, "batch dim"]:
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

def attraction(x:Float, R:Float, Z_alpha:int, Z_beta:int, Ne: int) -> Float[Array, "batch dim"]:
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
def exchange_correlation_one_dimensional(den:Float, Ne:int) -> Float[Array, "batch dim"]:
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