classdef Ref < handle
    properties
        Value;
    end
    methods
        function obj = Ref(value)
            if nargin == 1
                obj.Value = value;
            end
        end
    end
end
