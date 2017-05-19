function [m] = pyimport_proxy(varargin)
%pyimport_proxy Create a MATLAB proxy for viewing into Python module.
% @note Do NOT use on large modules!
m = PyProxy.fromPyValue(pyimport(varargin{:}));
end
