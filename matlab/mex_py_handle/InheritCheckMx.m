classdef InheritCheckMx < PyMxRaw
    methods
        function obj = InheritCheckMx()
            mod = pyimport_proxy('inherit_check_py');
            obj@PyMxRaw(mod.PyMxExtend);
        end
        
        function out = pure(~, value)
            out = sprintf('ml.pure=%s', value);
        end
        function out = optional(~, value)
            out = sprintf('ml.optional=%s', value);
        end
        
        % How to handle non-virtual methods?
        function out = dispatch(obj, value)
            out = obj.pyInvokeDirect('dispatch', value);
        end
    end
end
