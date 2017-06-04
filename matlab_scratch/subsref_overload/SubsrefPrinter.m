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
        
        function c = uminus(b)
            % Arriving here won't call subsref
            disp('neg');
            % However, this call will
            c = -b.Value;
        end
    end
end
