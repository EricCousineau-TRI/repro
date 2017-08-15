classdef weakref < handle
    properties
        Listener
        
        % Debugging
        Ref
        MetaListener
    end
    methods
        function obj = weakref(ref, storeRef)
            assert(isa(ref, 'handle'));
            % How to implement a weak reference?
            obj.Listener = event.listener(ref, 'ObjectBeingDestroyed', ...
                @(src, data) fprintf('weak ref: original destroyed\n'));
            % For debugging, store a direct reference.
            if storeRef
                obj.Ref = ref;
            end
%             % Track lifetime of listener.
%             obj.MetaListener = event.listener(obj.Listener, ...
%                 'ObjectBeingDestroyed', ...
%                 @(src, data) fprintf('Listener destroyed\n'));
        end
        function ref = get(obj)
            % What happens if the source goes out of scope?
            ref = obj.Listener.Source{1};
        end
    end
end
