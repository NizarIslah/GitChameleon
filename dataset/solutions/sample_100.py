# library: django
# version: 4.0.0
# extra_dependencies: []
import django
from django.conf import settings
from django.utils import timezone

settings.configure()


def get_time_in_utc(year: int, month: int, day: int) -> timezone.datetime:
    return timezone.datetime(year, month, day, tzinfo=timezone.utc)
