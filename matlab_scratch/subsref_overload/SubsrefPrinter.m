classdef SubsrefPrinter
% Goal: Check the totality of subsref
    properties
        Value
    end
    
    methods
        function obj = SubsrefPrinter(value)
            obj.Value = value;
        end
        
        function r = subsref(obj, s)
            disp('subsref:');
            disp(indent(yaml_dump(s), '  '));
            r = obj.Value;
        end
    end
end
