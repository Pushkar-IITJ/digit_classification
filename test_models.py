# Q3 solution:

import unittest
import joblib
import os
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModelTest(unittest.TestCase):
    def setUp(self):
        self.models_dir = 'models'
        self.roll_no = "M22AIE213"

    def test_model_type(self):
        for solver in ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']:
            model_path = os.path.join(self.models_dir, f"{self.roll_no}_lr_{solver}.joblib")
            with self.subTest(solver=solver):
                model = joblib.load(model_path)
                self.assertIsInstance(model, LogisticRegression)

    def test_solver_match(self):
        for solver in ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']:
            model_path = os.path.join(self.models_dir, f"{self.roll_no}_lr_{solver}.joblib")
            with self.subTest(solver=solver):
                model = joblib.load(model_path)
                self.assertEqual(model.get_params()['solver'], solver)

if __name__ == '__main__':
    unittest.main()