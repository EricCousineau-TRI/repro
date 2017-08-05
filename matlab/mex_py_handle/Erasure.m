classdef Erasure < handle
    properties (Access = protected)
        % Stored values.
        Values
        % Permit storing [] values, so use an external sentinel.
        References
    end
    
    methods
        function obj = Erasure()
            obj.Values = cell(1, 0);
            obj.References = zeros(1, 0);
            obj.resize(2);
        end
        
        function [i] = store(obj, value)
            i = find(obj.References == 0, 1, 'first');
            if isempty(i)
                i = obj.size() + 1;
                obj.resize(obj.size() + 4);
            end
            assert(isscalar(i));
            assert(isempty(obj.Values{i}));
            assert(obj.References(i) == 0);
            obj.Values{i} = value;
            obj.References(i) = 1;
            fprintf('ml: Store %d -> %d\n', i, obj.References(i));
            i = uint64(i);
        end
        
        function [value] = reference(obj, i)
            % Can only reference an existing object.
            assert(obj.References(i) > 0);
            obj.References(i) = obj.References(i) + 1;
            fprintf('ml: Ref %d -> %d\n', i, obj.References(i));
        end
        
        function [value] = dereference(obj, i)
            assert(i >= 1 && i <= obj.size());
            assert(obj.References(i) > 0);
            value = obj.Values{i};
            obj.References(i) = obj.References(i) - 1;
            fprintf('ml: Deref %d -> %d\n', i, obj.References(i));
            if obj.References(i) == 0
                % Clear cell, release reference.
                obj.Values{i} = [];
                fprintf('ml: Delete %d\n', i);
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
            obj.References(new_indices) = false;
        end
    end
end
