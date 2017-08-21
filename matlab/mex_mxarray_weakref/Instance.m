classdef Instance < handle
    properties
        Value
    end
    methods
        function obj = Instance(value)
            obj.Value = value;
        end
        function disp(obj)
            disp(obj.Value);
        end
    end
end
