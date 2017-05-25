classdef SubsrefPrinter
% Goal: Check the totality of subsref
% @ref https://www.mathworks.com/help/matlab/customize-object-indexing.html
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
        
        function ind = end(obj, subscriptIndex, subscriptCount)
            % For uniqueness
            ind = subscriptIndex * 10 + subscriptCount;
        end
    end
end
