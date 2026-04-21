import math
import random
import time
import numpy as np


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** (-1)

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "power must be a scalar number"
        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += (power * (self.data ** (power - 1))) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        x = max(-20, min(20, x))
        t = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class TensorValue:
    def __init__(self, data, _children=(), _op=''):
        if isinstance(data, (list, np.ndarray)):
            self.data = np.array(data, dtype=np.float64)
        else:
            self.data = data
        self.grad = np.zeros_like(self.data, dtype=np.float64) if isinstance(self.data, np.ndarray) else 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape if isinstance(self.data, np.ndarray) else ()

    def __repr__(self):
        return f"TensorValue(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, TensorValue) else TensorValue(np.full_like(self.data, other) if isinstance(self.data, np.ndarray) else other)
        out_data = self.data + other.data
        out = TensorValue(out_data, (self, other), '+')

        def _backward():
            g = out.grad
            self.grad += _unbroadcast(g, self.data.shape)
            other.grad += _unbroadcast(g, other.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, TensorValue) else TensorValue(np.full_like(self.data, other) if isinstance(self.data, np.ndarray) else other)
        out_data = self.data * other.data
        out = TensorValue(out_data, (self, other), '*')

        def _backward():
            g = out.grad
            self.grad += _unbroadcast(g * other.data, self.data.shape)
            other.grad += _unbroadcast(g * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        other = other if isinstance(other, TensorValue) else TensorValue(np.full_like(self.data, other) if isinstance(self.data, np.ndarray) else other)
        out_data = self.data / other.data
        out = TensorValue(out_data, (self, other), '/')

        def _backward():
            g = out.grad
            self.grad += _unbroadcast(g / other.data, self.data.shape)
            other.grad += _unbroadcast(-g * self.data / (other.data ** 2), other.data.shape)
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return TensorValue(np.full_like(self.data, other) if isinstance(self.data, np.ndarray) else other) / self

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out_data = self.data ** power
        out = TensorValue(out_data, (self,), f'**{power}')

        def _backward():
            g = out.grad
            self.grad += _unbroadcast(g * power * (self.data ** (power - 1)), self.data.shape)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = TensorValue(out_data, (self,), 'sum')

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis=axis)
            expanded = np.broadcast_to(g, self.data.shape)
            self.grad += expanded
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out_data = self.data.reshape(*shape)
        out = TensorValue(out_data, (self,), 'reshape')

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def exp(self):
        out_data = np.exp(self.data)
        out = TensorValue(out_data, (self,), 'exp')

        def _backward():
            self.grad += out.grad * out_data
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = TensorValue(t, (self,), 'tanh')

        def _backward():
            self.grad += out.grad * (1 - t ** 2)
        out._backward = _backward
        return out

    def relu(self):
        out_data = np.maximum(0, self.data)
        out = TensorValue(out_data, (self,), 'relu')

        def _backward():
            self.grad += out.grad * (self.data > 0).astype(np.float64)
        out._backward = _backward
        return out

    @staticmethod
    def matmul(a, b):
        out_data = np.matmul(a.data, b.data)
        out = TensorValue(out_data, (a, b), 'matmul')

        def _backward():
            g = out.grad
            if a.data.ndim == 1 and b.data.ndim == 1:
                a.grad += g * b.data
                b.grad += g * a.data
            elif a.data.ndim == 2 and b.data.ndim == 1:
                a.grad += np.outer(g, b.data)
                b.grad += a.data.T @ g
            elif a.data.ndim == 1 and b.data.ndim == 2:
                a.grad += g @ b.data.T
                b.grad += np.outer(a.data, g)
            else:
                a.grad += np.matmul(g, np.swapaxes(b.data, -2, -1))
                b.grad += np.matmul(np.swapaxes(a.data, -2, -1), g)
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        if isinstance(self.grad, np.ndarray):
            self.grad = np.ones_like(self.data)
        else:
            self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def _unbroadcast(grad, target_shape):
    if grad.shape == target_shape:
        return grad
    if target_shape == ():
        return np.sum(grad)
    ndim_diff = grad.ndim - len(target_shape)
    for _ in range(ndim_diff):
        grad = grad.sum(axis=0)
    for i, s in enumerate(target_shape):
        if s == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((xi * wi for xi, wi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.w + [neuron.b]]


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def numerical_gradient(f, *args, eps=1e-7):
    grads = []
    for arg_idx, arg in enumerate(args):
        if isinstance(arg, Value):
            old = arg.data
            arg.data = old + eps
            fp = f(*args).data
            arg.data = old - eps
            fm = f(*args).data
            arg.data = old
            grads.append((fp - fm) / (2 * eps))
        elif isinstance(arg, TensorValue):
            grad = np.zeros_like(arg.data)
            it = np.nditer(arg.data, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                old = arg.data[idx]
                arg.data[idx] = old + eps
                fp = f(*args)
                fp_val = fp.data if isinstance(fp, (Value, TensorValue)) else fp
                arg.data[idx] = old - eps
                fm = f(*args)
                fm_val = fm.data if isinstance(fm, (Value, TensorValue)) else fm
                arg.data[idx] = old
                grad[idx] = (fp_val - fm_val) / (2 * eps)
                it.iternext()
            grads.append(grad)
        else:
            grads.append(0.0)
    return grads


def check_gradients(f, *args, eps=1e-7, atol=1e-5, rtol=1e-3, verbose=True):
    out = f(*args)
    if isinstance(out, Value):
        out.backward()
        num_grads = numerical_gradient(f, *args, eps=eps)
        results = []
        for i, (arg, ng) in enumerate(zip(args, num_grads)):
            if isinstance(arg, Value):
                diff = abs(arg.grad - ng)
                max_diff = max(diff, 1e-12)
                rel_diff = diff / (abs(arg.grad) + abs(ng) + 1e-12)
                ok = diff < atol and rel_diff < rtol
                results.append(ok)
                if verbose:
                    status = "PASS" if ok else "FAIL"
                    print(f"  arg[{i}] analytical={arg.grad:.8f} numerical={ng:.8f} diff={diff:.2e} [{status}]")
            elif isinstance(arg, TensorValue):
                diff = np.abs(arg.grad - ng)
                max_diff = np.max(diff)
                rel_diff = diff / (np.abs(arg.grad) + np.abs(ng) + 1e-12)
                max_rel = np.max(rel_diff)
                ok = max_diff < atol and max_rel < rtol
                results.append(ok)
                if verbose:
                    status = "PASS" if ok else "FAIL"
                    print(f"  arg[{i}] max_diff={max_diff:.2e} max_rel_diff={max_rel:.2e} [{status}]")
        all_ok = all(results)
        if verbose:
            print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
        return all_ok
    elif isinstance(out, TensorValue):
        out.backward()
        num_grads = numerical_gradient(f, *args, eps=eps)
        results = []
        for i, (arg, ng) in enumerate(zip(args, num_grads)):
            if isinstance(arg, (Value, TensorValue)):
                ag = arg.grad
                diff = np.abs(ag - ng) if isinstance(ag, np.ndarray) else abs(ag - ng)
                max_diff = np.max(diff) if isinstance(diff, np.ndarray) else diff
                denom = (np.abs(ag) + np.abs(ng) + 1e-12) if isinstance(ag, np.ndarray) else (abs(ag) + abs(ng) + 1e-12)
                rel_diff = diff / denom
                max_rel = np.max(rel_diff) if isinstance(rel_diff, np.ndarray) else rel_diff
                ok = max_diff < atol and max_rel < rtol
                results.append(ok)
                if verbose:
                    status = "PASS" if ok else "FAIL"
                    print(f"  arg[{i}] max_diff={max_diff:.2e} max_rel_diff={max_rel:.2e} [{status}]")
        all_ok = all(results)
        if verbose:
            print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
        return all_ok


def benchmark_vs_pytorch(f_micro, f_torch, inputs_micro, inputs_torch, name="op", warmup=5, runs=100):
    import torch

    for _ in range(warmup):
        out = f_micro(*inputs_micro)
        if isinstance(out, Value):
            out.backward()
        elif isinstance(out, TensorValue):
            out.backward()
    for p in inputs_micro:
        if isinstance(p, Value):
            p.grad = 0.0
        elif isinstance(p, TensorValue):
            p.grad = np.zeros_like(p.data)

    t0 = time.perf_counter()
    for _ in range(runs):
        for p in inputs_micro:
            if isinstance(p, Value):
                p.grad = 0.0
            elif isinstance(p, TensorValue):
                p.grad = np.zeros_like(p.data)
        out = f_micro(*inputs_micro)
        if isinstance(out, Value):
            out.backward()
        elif isinstance(out, TensorValue):
            out.backward()
    t_micro = (time.perf_counter() - t0) / runs

    torch_inputs = []
    for inp in inputs_torch:
        if isinstance(inp, torch.Tensor):
            torch_inputs.append(inp.detach().requires_grad_(True))
        else:
            torch_inputs.append(inp)
    for _ in range(warmup):
        out = f_torch(*torch_inputs)
        if isinstance(out, torch.Tensor):
            out.backward()
    for inp in torch_inputs:
        if isinstance(inp, torch.Tensor):
            inp.grad = None

    t0 = time.perf_counter()
    for _ in range(runs):
        for inp in torch_inputs:
            if isinstance(inp, torch.Tensor):
                inp.grad = None
        out = f_torch(*torch_inputs)
        if isinstance(out, torch.Tensor):
            out.backward()
    t_torch = (time.perf_counter() - t0) / runs

    print(f"[{name}] micrograd: {t_micro * 1e6:.1f} µs | pytorch: {t_torch * 1e6:.1f} µs | ratio: {t_micro / t_torch:.1f}x")
    return t_micro, t_torch


if __name__ == "__main__":
    print("=== Scalar Value gradient check ===")
    a = Value(2.0)
    b = Value(-3.0)
    check_gradients(lambda a, b: a * b + a ** 2 - b, a, b)

    print("\n=== TensorValue gradient check ===")
    x = TensorValue(np.array([[1.0, 2.0], [3.0, 4.0]]))
    w = TensorValue(np.array([[0.5, -0.5], [1.0, 0.5]]))
    check_gradients(lambda x, w: TensorValue.matmul(x, w).sum(), x, w)

    print("\n=== Benchmark vs PyTorch ===")
    try:
        import torch

        def micro_fn(x, w):
            return TensorValue.matmul(x, w).sum()

        def torch_fn(x, w):
            return (x @ w).sum()

        x_np = np.random.randn(64, 64)
        w_np = np.random.randn(64, 64)
        benchmark_vs_pytorch(
            micro_fn, torch_fn,
            [TensorValue(x_np), TensorValue(w_np)],
            [torch.tensor(x_np, dtype=torch.float64), torch.tensor(w_np, dtype=torch.float64)],
            name="matmul(64x64 @ 64x64)"
        )

        def micro_scalar(a, b):
            return a * b + a.tanh() - b

        def torch_scalar(a, b):
            return a * b + torch.tanh(a) - b

        benchmark_vs_pytorch(
            micro_scalar, torch_scalar,
            [Value(3.0), Value(-1.5)],
            [torch.tensor(3.0, dtype=torch.float64, requires_grad=True),
             torch.tensor(-1.5, dtype=torch.float64, requires_grad=True)],
            name="scalar_ops"
        )
    except ImportError:
        print("PyTorch not installed, skipping benchmark")
