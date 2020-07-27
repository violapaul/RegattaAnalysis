
from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


yaml_path = '/Users/viola/tmp/foo.yml'
with open(yaml_path, 'r') as yaml_stream:
    data = load(yaml_stream, Loader=Loader)    
    
print(data)


