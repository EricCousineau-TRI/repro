function [m] = pyimport_proxy(module_name)
%pyimport_proxy Create a MATLAB proxy for viewing into Python module.
m = py_proxy(pyimport(module_name));
end
