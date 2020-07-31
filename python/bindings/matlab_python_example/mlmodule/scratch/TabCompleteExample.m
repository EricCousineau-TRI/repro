classdef TabCompleteExample < handle
    properties
        data
    end
    
    methods
        function obj = TabCompleteExample()
            obj.data = {'a', 'b', 'c'};
        end
        
        function p = properties(obj)
            p = obj.data;
        end
    end
end
