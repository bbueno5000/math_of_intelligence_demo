"""
The optimal values of m and b can be calculated with less effort doing a linear regression. 
This is to demonstrate gradient descent.
"""
from numpy import *

class MathOfIntelligence:
    """
    TODO: docstring
    """
    def __call__(self):
        """
        TODO: docstring
        """
        points = genfromtxt("data.csv", delimiter=",")
        learning_rate = 0.0001
        initial_b = 0 # initial y-intercept guess
        initial_m = 0 # initial slope guess
        num_iterations = 1000
        print('Starting gradient descent at b = {0}, m = {1}, error = {2}').format(
            initial_b, initial_m, self.compute_error_for_line_given_points(initial_b, initial_m, points))
        print('Running...')
        [b, m] = self.gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
        print('After {0} iterations b = {1}, m = {2}, error = {3}').format(
            num_iterations, b, m, self.compute_error_for_line_given_points(b, m, points))

    def compute_error_for_line_given_points(self, b, m, points):
        """
        y = mx + b
        m is slope, b is y-intercept
        """
        totalError = 0
        for i in range(0, len(points)):
            x = points[i, 0]
            y = points[i, 1]
            totalError += (y - (m * x + b)) ** 2
        return totalError / float(len(points))

    def gradient_descent_runner(self, points, starting_b, starting_m, learning_rate, num_iterations):
        """
        TODO: docstring
        """
        b = starting_b
        m = starting_m
        for i in range(num_iterations):
            b, m = self.step_gradient(b, m, array(points), learning_rate)
        return [b, m]

    def step_gradient(self, b_current, m_current, points, learningRate):
        """
        TODO: docstring
        """
        b_gradient = 0
        m_gradient = 0
        N = float(len(points))
        for i in range(0, len(points)):
            x = points[i, 0]
            y = points[i, 1]
            b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
            m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
        new_b = b_current - (learningRate * b_gradient)
        new_m = m_current - (learningRate * m_gradient)
        return [new_b, new_m]

if __name__ == '__main__':
    math_of_intelligence = MathOfIntelligence()
    math_of_intelligence()
