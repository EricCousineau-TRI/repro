classdef Obj < handle
    properties
        Value
    end
    methods
        function obj = Obj(value)
            obj.Value = value;
        end
        function disp(obj)
            disp(obj.Value);
        end
    end
end
