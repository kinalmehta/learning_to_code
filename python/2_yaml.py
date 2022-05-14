import yaml
from dotmap import DotMap

with open("./files/test.yaml", "r") as stream:
    try:
        args = DotMap(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

print(args.model)
