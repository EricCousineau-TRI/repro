classdef weakref < handle
    properties
        Ref
    end
    methods
        function obj = weakref(ref)
            assert(isa(ref, 'handle'));
            % How to implement a weak reference?
            % Attach a listener?
            obj.Ref = ref;
        end
        function ref = get(obj)
            ref = obj.Ref;
        end
    end
end
