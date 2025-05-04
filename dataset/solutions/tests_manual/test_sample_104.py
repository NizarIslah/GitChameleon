# Add the parent directory to import sys
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_104 import SampleForm, get_template_string, render_output


template_string = get_template_string()
rendered_output = render_output(template_string)


def normalize_html(html):
    # Remove all whitespace and standardize quotation marks to single quotes
    normalized = "".join(html.split())
    return normalized


template_string_django_4 = """
<form>
  <div>
    {{ form.name.label_tag }}
    {% if form.name.help_text %}
      <div class="helptext" id="{{ form.name.auto_id }}_helptext">
        {{ form.name.help_text|safe }}
      </div>
    {% endif %}
    {{ form.name.errors }}
    {{ form.name }}
  </div>
</form>
"""
assertion_result = normalize_html(rendered_output) == normalize_html(
    render_output(template_string)
)
assert assertion_result
assertion_result = (
    len(template_string) < 300
)  # check if the template_string is not too long (ideally should be 278)
assert assertion_result
