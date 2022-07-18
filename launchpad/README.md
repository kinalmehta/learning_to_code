
# launchpad from deepmind

Distributed computing framework.

### Points for using launchpad
- when making a function call to a node, the parameters should only be python objects and not numpy arrays or anything else
   - though it does seem to support returning numpy arrays
   - check file `4_*.py` and `5_*.py` to understand this

## Documentation

### Batching on a CourierNode
- Refer the following decorator
   - `from launchpad.nodes.courier.courier_utils import batched_handler`


## References
Library [link](https://github.com/deepmind/launchpad)<br>
Documentation [link](https://github.com/deepmind/launchpad/blob/master/docs/get_started.md)
