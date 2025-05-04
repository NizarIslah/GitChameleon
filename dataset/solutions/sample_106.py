# library: django
# version: 5.0.0
# extra_dependencies: []
import django
from django.conf import settings
from django.db import models, connection
from django.db.models import F

if not settings.configured:
    settings.configure(
        DATABASES={
            "default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}
        },
    )
django.setup()


def display_side_and_area(square):
    return square.side, square.area


def create_square(side):
    square = Square.objects.create(side=side)
    square.refresh_from_db()
    return square


class Square(models.Model):
    class Meta:
        app_label = "myapp"

    side = models.IntegerField()
    area = models.GeneratedField(
        expression=F("side") * F("side"),
        output_field=models.BigIntegerField(),
        db_persist=True,
    )
