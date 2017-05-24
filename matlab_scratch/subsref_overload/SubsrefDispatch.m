classdef SubsrefDispatch
% Goal: Only intercept top-level subsref stuff
    properties
        Value
    end
    
    methods
        function obj = SubsrefDispatch(value)
            obj.Value = value;
        end
        
        function y = doStuff(obj, x)
            if nargin < 2
                x = 100;
            end
            y = struct('out', obj.Value + x);
        end
        
        function r = subsref(obj, S)
            disp('subsref call:');
            disp(indent(yaml_dump(S), '  '));
%             remaining_start = 1;
            n = length(S);
            s = S(1);
            switch s.type
                case '.'
                    switch s.subs
                        case 'Value' % Property
                            % Dispatch to Value
                            r = subsref_relax(obj.Value, S(2:end));
                            return;
                        case 'doStuff' % Method
                            % Dispatch to doStuff
                            % If direct access, call as function
                            if n == 1
                                r = obj.doStuff();
                            else
                                sn = S(2);
                                assert(strcmp(sn.type, '()'));
                                r = obj.doStuff(sn.subs{:});
                                % Return following items
                                r = subsref_relax(r, S(3:end));
                            end
                            return;
                    end
            end
            disp('subsref unhandled:')
            disp(indent(yaml_dump(s), '  '));
            r = [];
        end        
    end
end

function r = subsref_relax(obj, S)
if isempty(S)
    r = obj;
else
    r = subsref(obj, S);
end
end
