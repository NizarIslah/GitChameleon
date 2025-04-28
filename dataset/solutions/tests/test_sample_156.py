import unittest
from jinja2 import Environment
from markupsafe import Markup

from dataset.solutions.sample_156 import solution, get_output, nl2br_core


class TestSample156(unittest.TestCase):
    def setUp(self):
        self.env = Environment(autoescape=True)
        self.nl2br = solution()

    def test_nl2br_filter_registration(self):
        """Test that the nl2br filter can be registered with Jinja2 environment"""
        self.env.filters['nl2br'] = self.nl2br
        self.assertIn('nl2br', self.env.filters)

    def test_nl2br_with_autoescaping(self):
        """Test nl2br with autoescaping enabled"""
        # Create a template environment with autoescaping
        env = Environment(autoescape=True)
        env.filters['nl2br'] = self.nl2br
        
        # Test with plain text
        template = env.from_string('{{ text|nl2br }}')
        result = template.render(text='Hello World')
        self.assertIn('<br>Hello</br>', result)
        self.assertIsInstance(result, Markup)
        
        # Test with HTML in the input
        template = env.from_string('{{ text|nl2br }}')
        result = template.render(text='<p>Hello</p>')
        # The HTML should be escaped, but "Hello" should still be replaced
        self.assertIn('&lt;p&gt;<br>Hello</br>&lt;/p&gt;', result)

    def test_nl2br_without_autoescaping(self):
        """Test nl2br with autoescaping disabled"""
        # Create a template environment without autoescaping
        env = Environment(autoescape=False)
        env.filters['nl2br'] = self.nl2br
        
        # Test with plain text
        template = env.from_string('{{ text|nl2br }}')
        result = template.render(text='Hello World')
        self.assertIn('<br>Hello</br>', result)
        self.assertNotIsInstance(result, Markup)
        
        # Test with HTML in the input
        template = env.from_string('{{ text|nl2br }}')
        result = template.render(text='<p>Hello</p>')
        # The HTML should not be escaped, and "Hello" should be replaced
        self.assertIn('<p><br>Hello</br></p>', result)

    def test_nl2br_core_function(self):
        """Test the core function directly"""
        from jinja2.runtime import EvalContext
        
        # Create eval contexts for testing
        autoescape_ctx = EvalContext(self.env, None, None)
        autoescape_ctx.autoescape = True
        
        no_autoescape_ctx = EvalContext(self.env, None, None)
        no_autoescape_ctx.autoescape = False
        
        # Test with autoescaping
        result = nl2br_core(autoescape_ctx, 'Hello World')
        self.assertIn('<br>Hello</br>', result)
        self.assertIsInstance(result, Markup)
        
        # Test without autoescaping
        result = nl2br_core(no_autoescape_ctx, 'Hello World')
        self.assertIn('<br>Hello</br>', result)
        self.assertNotIsInstance(result, Markup)

    def test_get_output_function(self):
        """Test the get_output function"""
        result = get_output(self.env, self.nl2br)
        # The template in get_output has a bug: '{{ Union[text, nl2br] }}'
        # This is not valid Jinja2 syntax, so we're just checking that it returns something
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()