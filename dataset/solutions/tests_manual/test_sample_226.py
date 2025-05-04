import pytest
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_226

import pluggy


def test_hookimpl_configuration_with_plugin_manager():
    pm = pluggy.PluginManager("pytest")

    class DummyPlugin:
        pytest_runtest_call = pytest_runtest_call

    plugin = DummyPlugin()
    pm.register(plugin)

    hookimpls = pm.hook.pytest_runtest_call.get_hookimpls()

    for impl in hookimpls:
        if impl.plugin is plugin:
            opts = impl.opts
            assert opts.get("tryfirst") is False
            break
    else:
        pytest.fail("pytest_runtest_call implementation not found in plugin manager.")


test_hookimpl_configuration_with_plugin_manager()
