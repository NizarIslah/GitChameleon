# library: django
# version: 5.0.0
# extra_dependencies: []
from django.conf import settings
from django.forms.models import BaseModelFormSet
from django.forms.renderers import get_default_renderer
from django.forms import Form

if not settings.configured:
    settings.configure()


def save_existing(formset: BaseModelFormSet, form: Form, instance: str) -> None:
    return formset.save_existing(form=form, obj=instance)
