# Add the parent directory to import sys
import os
import sys
import unittest
from unittest.mock import Mock, patch

from django.forms import Form
from django.forms.models import BaseModelFormSet

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_102 import save_existing


class DummyForm:
    def save(self, commit=True):
        return "dummy_instance_value_result"


class MyFormSet(BaseModelFormSet):
    def __init__(self, *args, **kwargs):
        self.renderer = get_default_renderer()
        super().__init__(*args, **kwargs)


fs5 = MyFormSet(queryset=[])
result = save_existing(formset=fs5, form=DummyForm(), instance="dummy_str")
assertion_result = result == "dummy_instance_value_result"
assert assertion_result
