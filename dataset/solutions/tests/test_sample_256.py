import unittest
from falcon import Request
from falcon.util.structures import Context

# Import the function to test
from dataset.solutions.sample_256 import custom_set_context


class TestCustomSetContext(unittest.TestCase):
    def setUp(self):
        # Create a new request object for each test
        self.req = Request(
            env={
                'REQUEST_METHOD': 'GET',
                'PATH_INFO': '/',
                'QUERY_STRING': '',
                'wsgi.input': None,
            }
        )

    def test_custom_set_context_sets_values(self):
        # Test that the function sets the role and user correctly
        role = "admin"
        user = "john_doe"
        
        context = custom_set_context(self.req, role, user)
        
        # Verify that the context has the correct values
        self.assertEqual(context.role, role)
        self.assertEqual(context.user, user)
        
        # Also verify that the request's context was updated
        self.assertEqual(self.req.context.role, role)
        self.assertEqual(self.req.context.user, user)
    
    def test_custom_set_context_returns_context(self):
        # Test that the function returns the context object
        context = custom_set_context(self.req, "user", "jane_doe")
        
        # Verify that the returned object is the request's context
        self.assertIs(context, self.req.context)
    
    def test_custom_set_context_with_empty_values(self):
        # Test with empty strings
        role = ""
        user = ""
        
        context = custom_set_context(self.req, role, user)
        
        # Verify that empty strings are set correctly
        self.assertEqual(context.role, role)
        self.assertEqual(context.user, user)


if __name__ == '__main__':
    unittest.main()