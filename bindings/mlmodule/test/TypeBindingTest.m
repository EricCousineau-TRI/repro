classdef TypeBindingTest < matlab.unittest.TestCase
%TypeBindingTest Write a MATLAB analog of testTypeBinding.py, and try to
%  get this to pass.

    methods (Test)
        function testBasic(testCase)
            tb = pyimport_proxy('pymodule.typebinding');

            % Without proxy:
            % Classes do not match (py.long <-> int64)
            % (Trivial)
            
            obj = tb.SimpleType(1);
            testCase.verifyEqual(obj.value(), int64(1));
            obj.set_value(2);
            testCase.verifyEqual(obj.value(), int64(2));
        end

        function testFlexible(testCase)
            tb = pyimport_proxy('pymodule.typebinding');
            
            % Same case as above, without proxy.

            obj = tb.SimpleType(1);
            testCase.verifyEqual(obj.value(), int64(1));
            obj.set_value(2.);
            testCase.verifyEqual(obj.value(), int64(2));
            % Expect non-integral floating point values to throw error
            identifier = 'MATLAB:Python:PyException';
            
            bad_ctor = @() tb.SimpleType(1.5);
            testCase.verifyError(bad_ctor, identifier);
            bad_set = @() obj.set_value(1.5);
            testCase.verifyError(bad_set, identifier);
            bad_type = @() obj.set_value('bad');
            testCase.verifyError(bad_type, identifier);
        end
        
        function testNumpyBasic(testCase)
            tb = pyimport_proxy('pymodule.typebinding');
            
            % Without proxy:
            %  'MATLAB:Python:UnsupportedInputArraySizeError'
            %   Error using py.type/subsref
            %   Conversion of MATLAB 'double' to Python is only supported for 1-N vectors.
            value = [1; 2; 3];
            obj = tb.EigenType(value);
            testCase.verifyEqual(obj.value(), value);
            
            % Without proxy, need to convert from numpy.ndarray to double
            % (Trivial)
            value = 1;
            obj = tb.EigenType(value);
            testCase.verifyEqual(obj.value(), value);
        end
    end
end
