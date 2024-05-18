If you have problems importing functions from `steering` into your `scratch` dir, you may need to add the following to the top of your script:
```python
import os
import sys
sys.path.append(os.path.abspath('..'))
```
