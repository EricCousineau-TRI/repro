classdef TypeBindingTest < matlab.unittest.TestCase

    methods (Test)
        function testBasic(testCase)
            tb = getModule();

            obj = tb.SimpleType(1);
            testCase.verifyEqual(obj.value(), 1);
            obj.set_value(2);
            testCase.verifyEqual(obj.value(), 2);
        end

        function testFlexible(testCase)
            tb = getModule();

            obj = tb.SimpleType(1);
            testCase.verifyEqual(obj.value(), 1);
            obj.set_value(2.);
            testCase.verifyEqual(obj.value(), 2);
            % Expect non-integral floating point values to throw error
            identifier = 'MATLAB:Python:PyException';
            
            bad_ctor = @() tb.SimpleType(1.5);
            testCase.verifyError(bad_ctor, identifier);
            bad_set = @() obj.set_value(1.5);
            testCase.verifyError(bad_set, identifier);
            bad_type = @() obj.set_value('bad');
            testCase.verifyError(bad_type, identifier);
        end
    end
end

function [tb] = getModule()
pyimport = @(m) py.importlib.import_module(m);
pydrake = pyimport('pydrake');
tb = pydrake.typebinding;
end
