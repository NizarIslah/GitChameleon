import unittest
from jinja2.runtime import Context
from dataset.solutions.sample_153 import setup_environment, solution


class TestSample153(unittest.TestCase):
    def test_solution_function(self):
        # Get the greet filter function
        greet_filter = solution()
        
        # Create a mock context with default prefix
        mock_context = {'prefix': 'Hello'}
        ctx = Context(environment=None, name='test_template', blocks={})
        ctx.vars = mock_context
        
        # Test with default prefix
        result = greet_filter(ctx, 'World')
        self.assertEqual(result, 'Hello, World!')
        
        # Test with custom prefix
        mock_context['prefix'] = 'Hi'
        result = greet_filter(ctx, 'John')
        self.assertEqual(result, 'Hi, John!')
        
        # Test without prefix (should use default 'Hello')
        ctx.vars = {}
        result = greet_filter(ctx, 'Alice')
        self.assertEqual(result, 'Hello, Alice!')
    
    def test_setup_environment(self):
        # Get the greet filter function
        greet_filter = solution()
        
        # Setup environment with the filter
        env = setup_environment('greet', greet_filter)
        
        # Check if filter was registered correctly
        self.assertIn('greet', env.filters)
        self.assertEqual(env.filters['greet'], greet_filter)
        
        # Test the filter in a template
        template = env.from_string("{{ 'World' | greet }}")
        result = template.render(prefix="Welcome")
        self.assertEqual(result, 'Welcome, World!')
        
        # Test with different prefix
        result = template.render(prefix="Hola")
        self.assertEqual(result, 'Hola, World!')
        
        # Test without prefix (should use default 'Hello')
        result = template.render()
        self.assertEqual(result, 'Hello, World!')


if __name__ == '__main__':
    unittest.main()