classdef NumPyProxy < PyProxy
% Sliceable, castable NumPy proxy.
% A lot of stuff to do something simple, but it allows a *little* smoother
% integration with NumPy.

    methods
        function obj = NumPyProxy(p)
            if isnumeric(p)
                % Convert to nparray
                p = matpy.mat2nparray(p);
            end
            obj@PyProxy(p);
        end
        
        function out = isArithmetic(obj)
            p = PyProxy.getPy(obj);
            helper = pyimport('proxy_helper'); % Put on PYTHONPATH
            out = helper.np_is_arithemtic(p);
        end
    end
    
    methods
        function t = transpose(obj)
            p = PyProxy.getPy(obj);
            t = NumPyProxy(p.T);
        end
        
        function t = ctranspose(obj)
            % TODO: Return conjugate if dtype implies complex
            t = transpose(obj);
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
        
        function sz = size(obj, dim)
            p = PyProxy.getPy(obj);
            if nargin < 2
                sz = cellfun(@int64, cell(p.shape));
                if isscalar(sz)
                    % Make it look like a 2D vector... Is this a bad idea?
                    sz = [sz, int64(1)];
                end
            else
                sz = p.shape{dim};
            end
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
            % Retrieve something with a final substruct
            % https://stackoverflow.com/questions/2936863/python-implementing-slicing-in-getitem
            [pView, pKeys] = obj.pyGetSubViewAndKeys(s);
            get = py.getattr(pView, '__getitem__');
            pValue = get(pKeys);
            r = PyProxy.fromPyValue(pValue);
        end
        
        function pySubsasgn(obj, s, value)
            % Assign something with a final substruct
            [pView, pKeys] = obj.pyGetSubViewAndKeys(s);
            set = py.getattr(pView, '__setitem__');
            pValue = PyProxy.toPyValue(value);
            set(pKeys, pValue);
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
            % TODO: Preserve 2D shape of slices?
            pKeys = substruct_to_py_slice(s.subs);
        end
    end
end
