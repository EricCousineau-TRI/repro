classdef NumPyProxy < PyProxy
% Sliceable NumPy proxy
    methods
        function obj = NumPyProxy(p)
            obj@PyProxy(p);
        end
        
        function ind = end(obj, subscriptIndex, subscriptCount)
            p = PyProxy.getPy(obj);
            % Play naive
            if subscriptCount == 1
                % Present flattened count
                ind = p.size;
            else
                % Require same size
                assert(subscriptCount == p.ndim);
                ind = p.shape{subscriptIndex};
            end
        end
        
        function out = isArithmetic(obj)
            p = PyProxy.getPy(obj);
            helper = pyimport('proxy_helper'); % Put on PYTHONPATH
            out = helper.np_is_arithemtic(p);
        end
        
        function disp(obj)
            disp('  [NumPyProxy]');
            p = PyProxy.getPy(obj);
            disp(char(py.str(p))); % need indent
        end
        
        function out = double(obj)
            p = PyProxy.getPy(obj);
            out = matpy.nparray2mat(p);
        end
    end
    
    methods (Access = protected)
        function r = pySubsref(obj, s)
            % https://stackoverflow.com/questions/2936863/python-implementing-slicing-in-getitem
            p = PyProxy.getPy(obj);
            if length(s) == 1 && p.ndim > 1
                % Use flat view for 1D access to 2+D arrays
                view = p.flat;
            else
                view = p;
            end
            get = py.getattr(view, '__getitem__');
            pKey = obj.getPyKey(s);
            pValue = get(pKey);
            r = PyProxy.fromPyValue(pValue);
        end
        
        function pySubsasgn(obj, s, value)
            % Constructor indexing, either a py.slice or an index list
            error('Not implemented');
        end
        
        function pKeys = getPyKey(~, s)
            pKeys = substruct_to_py_slice(s.subs);
        end
    end
end
