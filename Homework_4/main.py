import os
import pandas as pd
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import em_algorithm as em
import naive_bayes as NB


def generate_r2_gaussian(alpha, samples, seed=73):
    """
    Generate 2 gaussian clusters based on the expected errors
    """

    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)

    # calculate Z and distance between clusters
    z = jstats.norm.ppf(1-alpha)
    d = 2 * z

    # define cluster parameters
    mu1 = jnp.array([0.0, 0.0])  # origin
    mu2 = jnp.array([d, 0.0])

    # Cluster generation (normal distribution). Generate a cluster with n samples moved to
    # the claculated distance over the plane
    cluster_1 = jax.random.normal(key1, (samples, 2)) + mu1
    cluster_2 = jax.random.normal(key2, (samples, 2)) + mu2

    # concatenate both clusters to create the ground truth
    X = jnp.vstack([cluster_1, cluster_2])

    # -- Validation --
    # generate labels
    labels_1 = jnp.zeros(samples)
    labels_2 = jnp.ones(samples)
    Y = jnp.concatenate([labels_1, labels_2])
    return X, Y


def main():
    alpha_c1 = 0.025
    alpha_c2 = 0.25
    samples = 500

    x_case1, y_case1 = generate_r2_gaussian(alpha_c1, samples)
    x_case2, y_case2 = generate_r2_gaussian(alpha_c2, samples)

    model_1 = em.em_algorithm()
    model_1.fit_em(x_case1)

    print("Parameters calculated by EM:")
    print("Means (mu):\n", model_1.mu)
    print("#"*50)
    print("Covariance (sigma):\n", model_1.sigma)

    pi1 = model_1.pi
    mu1 = model_1.mu
    sigma1 = model_1.sigma

    case1_NB = NB.naive_bayes(pi1, mu1, sigma1, x_case1)
    case1_NB.model()
    print(f"Predictions for case 1 is {case1_NB.y_pred}")

    ##########################################################################
    print("_"*65)

    model_2 = em.em_algorithm()
    model_2.fit_em(x_case2)

    print("Means (mu):\n", model_2.mu)
    print("#"*50)
    print("Covariance (sigma):\n", model_2.sigma)

    pi2 = model_2.pi
    mu2 = model_2.mu
    sigma2 = model_2.sigma

    case2_NB = NB.naive_bayes(pi2, mu2, sigma2, x_case2)
    case2_NB.model()
    print(f"Predictions for case 2 is {case2_NB.y_pred}")


if __name__ == "__main__":
    main()
