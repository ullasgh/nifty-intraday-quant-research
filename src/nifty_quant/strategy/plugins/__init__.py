"""Built-in strategy plugin implementations.

Importing this package imports every submodule in it, so that each module's
``@nifty_quant.strategy.registry.register``-decorated ``Strategy`` subclass runs
its decorator and registers itself. Without this, `registry.get(name)` would
raise ``KeyError`` for every built-in plugin even after `import
nifty_quant.strategy.plugins`, because the decorator never executes until its
module is imported.
"""

import importlib
import pkgutil

for _module_info in pkgutil.iter_modules(__path__, prefix=f"{__name__}."):
    importlib.import_module(_module_info.name)

del _module_info
