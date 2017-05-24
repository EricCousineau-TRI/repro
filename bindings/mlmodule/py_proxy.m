function [m] = py_proxy(p)
%py_proxy Convenience wrapper.
m = PyProxy.fromPyValue(p);
end
