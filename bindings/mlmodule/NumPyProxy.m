classdef NumPyProxy < PyProxy
% Sliceable NumPy proxy
    methods
        function obj = NumPyProxy(p)
            obj@PyProxy(p);
        end
        
        function t = transpose(obj)
            p = PyProxy.getPy(obj);
            t = NumPyProxy(p.T);
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
            if p.ndim == 1
                % Show vectors as a row
                view = p.view().reshape([-1, 1]); % err... somehow...
                disp('    1D');
            else
                view = p;
            end
            disp(char(py.str(view))); % need indent
        end
        
        % More generalized conversion mechanism???
        function out = double(obj)
            p = PyProxy.getPy(obj);
            out = matpy.nparray2mat(p);
        end
    end
    
    methods (Access = protected)
        function r = pySubsref(obj, s)
            % https://stackoverflow.com/questions/2936863/python-implementing-slicing-in-getitem
            [pView, pKeys] = obj.pyGetSubViewAndKeys(s);
            get = py.getattr(pView, '__getitem__');
            pValue = get(pKeys);
            r = PyProxy.fromPyValue(pValue);
        end
        
        function pySubsasgn(obj, s, value)
            % Constructor indexing, either a py.slice or an index list
            error('Not implemented');
        end
        
        function [pView, pKeys] = pyGetSubViewAndKeys(obj, s)
            p = PyProxy.getPy(obj);
            if length(s.subs) == 1 && p.ndim > 1
                % Use flat view for 1D access to 2+D arrays
                if p.flags.c_contiguous
                    % Stored in C order (row-major, column-contiguous).
                    pView = p.T.flat;
                else
                    pView = p.flat;
                end
            else
                pView = p;
            end
            pKeys = substruct_to_py_slice(s.subs);
        end
    end
end
