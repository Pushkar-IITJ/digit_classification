import unittest
from api import app

class FlaskApiTest(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_predict_svm(self):
        test_data = {'feature': ["Appropriate test values"]}
        response = self.app.post('/predict/svm', json=test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json)

    def test_predict_lr(self):
        test_data = {'feature': ["Appropriate test values"]}
        response = self.app.post('/predict/lr', json=test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json)

    def test_predict_tree(self):
        test_data = {'feature': ["Appropriate test values"]}
        response = self.app.post('/predict/tree', json=test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json)

if __name__ == '__main__':
    unittest.main()
