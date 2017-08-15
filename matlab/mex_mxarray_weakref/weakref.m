classdef weakref < handle
    properties
        Ref
        Listener
    end
    methods
        function obj = weakref(ref)
            assert(isa(ref, 'handle'));
            % How to implement a weak reference?
            % Attach a listener?
            obj.Ref = ref;
            obj.Listener = event.listener(ref, 'ObjectBeingDestroyed', ...
                @(varargin) obj.destroyed(varargin{:}));
        end
        function ref = get(obj)
            ref = obj.Ref;
%             ref = [];
        end
        function destroyed(obj, ref, eventData)
            fprintf('Destroyed\n');
        end
    end
end
