
import haiku as hk
import jax
import numpy as np
import jax.numpy as jnp


class VisualFeatures(hk.Module):
  """Simple convolutional stack commonly used for Atari."""

  def __init__(self):
    super().__init__(name='test_model')
    self._cnn = hk.Sequential([
        hk.Conv2D(16, [8, 8], 8, padding="VALID"),
        jax.nn.relu,
        hk.Conv2D(32, [4, 4], 1, padding="VALID"),
        jax.nn.relu,
    ])
    self._ff = hk.Sequential([
        hk.Linear(64),
        jax.nn.relu,
        hk.Linear(64),
        jax.nn.relu,
    ])

  def __call__(self, inputs) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)

    outputs = self._cnn(inputs)

    if batched_inputs:
      outputs =  jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    else:
        outputs =  jnp.reshape(outputs, [-1])  # [D]
    
    outputs = self._ff(outputs)
    return outputs


if __name__=="__main__":
    rng = jax.random.PRNGKey(42)
    ip = np.random.random((1,64,64,3)).astype(np.float32)
    ip_2 = np.random.random((64,64,3)).astype(np.float32)
    model_fn = lambda x: VisualFeatures()(x)
    model = hk.without_apply_rng(hk.transform(model_fn))
    params = model.init(rng, ip)

    res = model.apply(params, ip)
    print(res.shape)
    res2 = model.apply(params, ip_2)
    print(res2.shape)
