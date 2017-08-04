classdef Erasure < handle
    properties (Access = protected)
        % Stored values.
        Values
        % Permit storing [] values, so use an external sentinel.
        Occupied
    end
    
    methods
        function obj = Erasure()
            obj.Values = cell(1, 0);
            obj.Occupied = false(1, 0);
            obj.resize(2);
        end
        
        function [i] = store(obj, value)
            i = find(~obj.Occupied, 1, 'first');
            if isempty(i)
                i = obj.size() + 1;
                obj.resize(obj.size() + 4);
            end
            assert(isscalar(i));
            assert(isempty(obj.Values{i}));
            assert(~obj.Occupied(i));
            obj.Values{i} = value;
            obj.Occupied(i) = true;
            i = uint64(i);
        end
        
        function [value] = dereference(obj, i, keep)
            if nargin < 3
                keep = false;
            end
            assert(i >= 1 && i <= obj.size());
            assert(obj.Occupied(i));
            value = obj.Values{i};
            if ~keep
                % Clear cell
                obj.Values{i} = [];
                obj.Occupied(i) = false;
            end
        end
    end
    
    methods (Access = protected)
        function [sz] = size(obj)
            sz = length(obj.Values);
        end
        
        function [] = resize(obj, new_sz)
            sz = obj.size();
            assert(new_sz >= sz);
            dsz = new_sz - sz;
            new_indices = sz + 1:new_sz;
            obj.Values(new_indices) = cell(1, dsz);
            obj.Occupied(new_indices) = false;
        end
    end
end
