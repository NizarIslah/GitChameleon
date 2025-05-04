# library: django
# version: 4.0.0
# extra_dependencies: []
import django
from django.conf import settings
from django.db import models

if not settings.configured:
    settings.configure()
django.setup()

color = models.TextChoices("Color", "RED GREEN BLUE")


class MyModel(models.Model):
    class Meta:
        app_label = "myapp"

    color = models.CharField(max_length=5, choices=color.choices)
