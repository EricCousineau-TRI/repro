classdef weakref < handle
% How to implement a proper weak reference?
    properties (Access = private)
        Listener
    end

    methods
        function obj = weakref(ref)
            assert(isa(ref, 'handle'));
            % Attach a listener to attempt to make a weak reference, per
            % the lifetime indications in the documentation:
            % https://www.mathworks.com/help/releases/R2016b/matlab/matlab_oop/listener-lifecycle.html

            obj.Listener = event.listener(ref, 'ObjectBeingDestroyed', ...
                @(varargin) fprintf('weak ref: original destroyed (non-closure callback)\n'));
        end

        function ref = get(obj)
            disp(obj.Listener);
            if ~isempty(obj.Listener) && obj.Listener.Enabled
                ref = obj.Listener.Source{1};
                if ~isvalid(ref)
                    ref = [];
                end
            else
                ref = [];
            end
        end
    end
end
