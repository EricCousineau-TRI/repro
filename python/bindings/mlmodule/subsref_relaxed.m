function r = subsref_relaxed(obj, S)
if isempty(S)
    r = obj;
else
    r = subsref(obj, S);
end
end
