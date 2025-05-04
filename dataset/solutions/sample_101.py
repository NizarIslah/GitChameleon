# library: django
# version: 4.0.0
# extra_dependencies: []
from django.conf import settings
from django.forms.models import BaseModelFormSet
from django.forms.renderers import get_default_renderer
from django.forms import Form

if not settings.configured:
    settings.configure()


def save_existing(formset: BaseModelFormSet, form: Form, obj: str) -> None:
    return formset.save_existing(form=form, instance=obj)
