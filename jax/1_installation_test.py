import jax
import jaxlib
import jaxlib.xla_extension as xe

print(jax.random.PRNGKey(0))
print(jax.devices())
print(xe.CpuDevice)


