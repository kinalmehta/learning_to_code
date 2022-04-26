
import haiku as hk
import jax
import numpy as np
import jax.numpy as jnp

def test_one_hot():
    a=np.random.randint(0,10, size=(10))
    b=np.random.randint(0,10)
    hot_a = jax.nn.one_hot(a, 100)
    hot_b = jax.nn.one_hot(b, 100)
    print(hot_a.shape, a.shape)
    print(hot_b.shape, b)

if __name__=="__main__":
    test_one_hot()
