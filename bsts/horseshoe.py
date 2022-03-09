import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def horseshoe_prior(p):
    r_local = numpyro.sample('r_local', dist.Normal(jnp.zeros(p)))
    half = jnp.ones(p) * 0.5
    rho_local = numpyro.sample('rho_local', dist.InverseGamma(half, half))
    r_global = numpyro.sample('r_global', dist.Normal(0.0))
    rho_global = numpyro.sample('rho_global', dist.InverseGamma(0.5, 0.5))
    z = numpyro.sample('z', dist.Normal(1.0, 100.0))
    lam = r_local * jnp.sqrt(rho_local)
    tau = r_global * jnp.sqrt(rho_global)
    beta = numpyro.primitives.deterministic('beta', z * lam * tau)
    return beta
