classdef weakref < handle
% How to implement a proper weak reference?
    properties (Access = private)
        Ref
        Listener
        ListenerClosure
    end

    methods
        function obj = weakref(ref, storeRef)
            assert(isa(ref, 'handle'));
            % Attach a listener to attempt to make a weak reference, per
            % the lifetime indications in the documentation:
            % https://www.mathworks.com/help/releases/R2016b/matlab/matlab_oop/listener-lifecycle.html

            % - This does not get called.
            obj.Listener = event.listener(ref, 'ObjectBeingDestroyed', ...
                @(varargin) fprintf('weak ref: original destroyed (non-closure callback)\n'));
            % - This does get called, when storeRef is true.
            obj.ListenerClosure = event.listener(ref, 'ObjectBeingDestroyed', ...
                @(varargin) obj.destroyed(varargin{:}));
            % Store reference conditionally.
            if storeRef
                obj.Ref = ref;
            end
        end

        function ref = get(obj)
            ref = obj.Listener.Source{1};
        end
    end

    methods (Access = private)
        function destroyed(obj, ref, eventData) %#ok<INUSD>
            fprintf('weak ref: original destroyed (closure callback)\n');
        end
    end
end
