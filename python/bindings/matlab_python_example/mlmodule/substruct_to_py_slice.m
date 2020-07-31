function pKeys = substruct_to_py_slice(subs)
% Construct subsctruct subscripts to index, slice, slice tuple, or ragged
% indexing
%
% Setup:
%   ':' -> py.slice()
%   array -> convert to int64
%   - if all of the array are evenly spaced, create slice
%     (inefficient, but meh).
%   - otherwise, convert to numpy array
%
% >> substruct_to_py_slice({1, 1:10, [1, 3, 7], ':'})
% 0
% slice(0L, 9L, 1L)
% 0   2   6
% slice(None, None, None)

assert(iscell(subs));
n = length(subs);
pKeys = cell(1, n);
for i = 1:n
    sub = subs{i};
    if ischar(sub)
        assert(strcmp(sub, ':'));
        pKey = py.slice(py.None);
    elseif isnumeric(sub)
        % TODO: Check to ensure that this is valid integral.
        sub = int64(sub);
        % Convert from 1-based to 0-based
        if isscalar(sub)
            pKey = sub - 1;
        else
            % Not sure if this is any better than just passing it through.
            % Meh.
            dsub = diff(sub);
            if all(dsub == dsub(1))
                % Apply extra bounds, 'cause Python
                fin = sub(end) - 1 + dsub(1);
                if fin < 0
                    % Need this to capture first element, if that's where a
                    % reverse sequence ends.
                    % NOTE: This should ignore ragged stuff.
                    fin = py.None;
                end
                pKey = py.slice(sub(1) - 1, fin, dsub(1));
            else
                % Meh. Just pass it along.
                pKey = sub - 1;
            end
        end
    elseif PyProxy.isPy(sub)
        % Python type. Pass through.
        pKey = sub;
    else
        error('Invalid subscript type');
    end
    pKeys{i} = pKey;
end

end
