classdef Erasure < handle
    properties (Access = protected)
        % Stored values.
        Values
        % Permit storing [] values, so use an external sentinel.
        References
        % Debugging
        Debug = true
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
            if obj.Debug
                fprintf('ml: Store %d -> %d\n', i, obj.References(i));
            end
            i = uint64(i);
        end
        
        function [] = incrementReference(obj, i)
            % Can only reference an existing object.
            assert(obj.References(i) > 0);
            obj.References(i) = obj.References(i) + 1;
            if obj.Debug
                fprintf('ml: Ref %d -> %d\n', i, obj.References(i));
            end
        end
        
        function [] = decrementReference(obj, i)
            obj.References(i) = obj.References(i) - 1;
            if obj.Debug
                fprintf('ml: Deref %d -> %d\n', i, obj.References(i));
            end
            if obj.References(i) == 0
                % Clear cell, release reference.
                obj.Values{i} = [];
                if obj.Debug
                    fprintf('ml: Delete %d\n', i);
                end
            end
        end
        
        function [value] = retrieve(obj, i)
            assert(obj.isValid(i));
            value = obj.Values{i};
        end
        
        function [n] = count(obj)
            % Return number of live references.
            n = nnz(obj.References);
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
        
        function [valid] = isValid(obj, i)
            valid = false;
            if i >= 1 && i <= obj.size()
                if obj.References(i) > 0
                    valid = true;
                end
            end
        end
    end
end
