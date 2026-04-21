import pytest
import numpy as np
from engine import Value, TensorValue, Neuron, Layer, MLP, numerical_gradient, check_gradients


class TestValue:
    def test_add(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0

    def test_mul(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c.data == 6.0

    def test_sub(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        assert c.data == 2.0

    def test_div(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        assert c.data == pytest.approx(2.0)

    def test_pow(self):
        a = Value(3.0)
        c = a ** 2
        assert c.data == 9.0

    def test_neg(self):
        a = Value(3.0)
        c = -a
        assert c.data == -3.0

    def test_exp(self):
        import math
        a = Value(1.0)
        c = a.exp()
        assert c.data == pytest.approx(math.e)

    def test_tanh(self):
        a = Value(0.0)
        c = a.tanh()
        assert c.data == pytest.approx(0.0)

    def test_relu(self):
        a = Value(2.0)
        assert a.relu().data == 2.0
        b = Value(-1.0)
        assert b.relu().data == 0.0

    def test_backward_add(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_backward_mul(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        assert a.grad == 3.0
        assert b.grad == 2.0

    def test_backward_complex(self):
        a = Value(2.0)
        b = Value(-3.0)
        c = a * b + a
        c.backward()
        assert a.grad == pytest.approx(-2.0)
        assert b.grad == pytest.approx(2.0)

    def test_gradient_accumulation(self):
        a = Value(3.0)
        b = a + a
        b.backward()
        assert a.grad == 2.0

    def test_rmul(self):
        a = Value(2.0)
        c = 3 * a
        assert c.data == 6.0

    def test_radd(self):
        a = Value(2.0)
        c = 3 + a
        assert c.data == 5.0

    def test_rsub(self):
        a = Value(2.0)
        c = 5 - a
        assert c.data == 3.0

    def test_scalar_add(self):
        a = Value(2.0)
        c = a + 3
        assert c.data == 5.0


class TestTensorValue:
    def test_add(self):
        a = TensorValue(np.array([1.0, 2.0]))
        b = TensorValue(np.array([3.0, 4.0]))
        c = a + b
        np.testing.assert_array_equal(c.data, [4.0, 6.0])

    def test_mul(self):
        a = TensorValue(np.array([1.0, 2.0]))
        b = TensorValue(np.array([3.0, 4.0]))
        c = a * b
        np.testing.assert_array_equal(c.data, [3.0, 8.0])

    def test_matmul_2d(self):
        a = TensorValue(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = TensorValue(np.array([[5.0, 6.0], [7.0, 8.0]]))
        c = TensorValue.matmul(a, b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_equal(c.data, expected)

    def test_matmul_2x1(self):
        a = TensorValue(np.array([[1.0, 2.0]]))
        b = TensorValue(np.array([[3.0], [4.0]]))
        c = TensorValue.matmul(a, b)
        np.testing.assert_array_equal(c.data, [[11.0]])

    def test_sum(self):
        a = TensorValue(np.array([[1.0, 2.0], [3.0, 4.0]]))
        s = a.sum()
        assert s.data == pytest.approx(10.0)

    def test_sum_axis(self):
        a = TensorValue(np.array([[1.0, 2.0], [3.0, 4.0]]))
        s = a.sum(axis=0)
        np.testing.assert_array_equal(s.data, [4.0, 6.0])

    def test_broadcasting_add(self):
        a = TensorValue(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = TensorValue(np.array([10.0, 20.0]))
        c = a + b
        expected = np.array([[11.0, 22.0], [13.0, 24.0]])
        np.testing.assert_array_equal(c.data, expected)

    def test_reshape(self):
        a = TensorValue(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = a.reshape(4)
        np.testing.assert_array_equal(b.data, [1.0, 2.0, 3.0, 4.0])

    def test_relu(self):
        a = TensorValue(np.array([-1.0, 0.0, 1.0, 2.0]))
        b = a.relu()
        np.testing.assert_array_equal(b.data, [0.0, 0.0, 1.0, 2.0])

    def test_exp(self):
        a = TensorValue(np.array([0.0, 1.0]))
        b = a.exp()
        np.testing.assert_array_almost_equal(b.data, [1.0, np.e])

    def test_tanh(self):
        a = TensorValue(np.array([0.0, 1.0]))
        b = a.tanh()
        np.testing.assert_array_almost_equal(b.data, np.tanh([0.0, 1.0]))

    def test_backward_add(self):
        a = TensorValue(np.array([1.0, 2.0]))
        b = TensorValue(np.array([3.0, 4.0]))
        c = a + b
        c.backward()
        np.testing.assert_array_equal(a.grad, [1.0, 1.0])
        np.testing.assert_array_equal(b.grad, [1.0, 1.0])

    def test_backward_matmul(self):
        a = TensorValue(np.array([[1.0, 2.0]]))
        b = TensorValue(np.array([[3.0], [4.0]]))
        c = TensorValue.matmul(a, b)
        s = c.sum()
        s.backward()
        np.testing.assert_array_equal(a.grad, [[3.0, 4.0]])
        np.testing.assert_array_equal(b.grad, [[1.0], [2.0]])

    def test_backward_broadcasting(self):
        a = TensorValue(np.array([[1.0, 2.0]]))
        b = TensorValue(np.array([[10.0, 20.0], [30.0, 40.0]]))
        c = a + b
        s = c.sum()
        s.backward()
        np.testing.assert_array_equal(a.grad, [[2.0, 2.0]])
        np.testing.assert_array_equal(b.grad, [[1.0, 1.0], [1.0, 1.0]])


class TestGradientChecker:
    def test_scalar_gradient_check(self):
        a = Value(2.0)
        b = Value(-3.0)
        assert check_gradients(lambda a, b: a * b + a ** 2 - b, a, b, verbose=False)

    def test_scalar_tanh_gradient(self):
        a = Value(1.5)
        assert check_gradients(lambda a: a.tanh(), a, verbose=False)

    def test_scalar_exp_gradient(self):
        a = Value(0.7)
        assert check_gradients(lambda a: a.exp(), a, verbose=False)

    def test_scalar_relu_gradient(self):
        a = Value(0.5)
        assert check_gradients(lambda a: a.relu(), a, verbose=False)

    def test_tensor_matmul_gradient(self):
        x = TensorValue(np.array([[1.0, 2.0], [3.0, 4.0]]))
        w = TensorValue(np.array([[0.5, -0.5], [1.0, 0.5]]))
        assert check_gradients(lambda x, w: TensorValue.matmul(x, w).sum(), x, w, verbose=False)

    def test_tensor_broadcast_gradient(self):
        a = TensorValue(np.array([[1.0, 2.0]]))
        b = TensorValue(np.array([[3.0, 4.0], [5.0, 6.0]]))
        assert check_gradients(lambda a, b: (a * b).sum(), a, b, verbose=False)

    def test_tensor_exp_gradient(self):
        a = TensorValue(np.array([0.5, -0.3, 1.2]))
        assert check_gradients(lambda a: a.exp().sum(), a, verbose=False)

    def test_tensor_relu_gradient(self):
        a = TensorValue(np.array([-1.0, 0.5, 2.0]))
        assert check_gradients(lambda a: a.relu().sum(), a, verbose=False)


class TestMLP:
    def test_neuron(self):
        n = Neuron(3)
        out = n([1.0, 2.0, 3.0])
        assert isinstance(out, Value)
        assert -1.0 <= out.data <= 1.0

    def test_layer(self):
        l = Layer(3, 2)
        out = l([1.0, 2.0, 3.0])
        assert isinstance(out, list)
        assert len(out) == 2

    def test_mlp(self):
        m = MLP(3, [4, 4, 1])
        out = m([1.0, 2.0, 3.0])
        assert isinstance(out, Value)

    def test_parameters(self):
        m = MLP(3, [4, 1])
        params = m.parameters()
        assert len(params) > 0
