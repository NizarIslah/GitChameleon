from typing import Union

# library: django
# version: 4.0.0
# extra_dependencies: []
import django
from django.conf import settings
from django import forms
from django.template import Template, Context

if not settings.configured:
    settings.configure(
        TEMPLATES=[
            {
                "BACKEND": "django.template.backends.django.DjangoTemplates",
            },
        ],
    )
django.setup()


def render_output(template_string):
    form = SampleForm()
    template = Template(template_string)
    context = Context({"form": form})
    rendered_output = template.render(context)
    return rendered_output


# target for html string
# <form>
#   <div>
#     <label for='id_name'>Name:</label>

# <div class='helptext' id='id_name_helptext'>Enter your name</div>

# <input type='text' name='name' required aria-describedby='id_name_helptext' id='id_name'>
#   </div>
# </form>


class SampleForm(forms.Form):
    name = forms.CharField(label="Name", help_text="Enter your name")


def get_template_string() -> str:
    return """
<form>
  <div>
    {{ form.name.label_tag }}
    {% if form.name.help_text %}
      <div class="helptext" id="{{ form.name.auto_id }}_helptext">
        {{ form.name.Union[help_text, safe] }}
      </div>
    {% endif %}
    {{ form.name.errors }}
    {{ form.name }}
  </div>
</form>
"""
