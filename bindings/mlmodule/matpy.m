classdef matpy
    %MATPY Summary of this class goes here
    %   Detailed explanation goes here
    % @ref https://www.mathworks.com/matlabcentral/answers/157347-convert-python-numpy-array-to-double#comment_437274
    % @author Contributors to this post.
    
    % TODO(eric.cousineau): Consider non-double nparray's, via nparray.dtype
    % Will this work for npmatrix?
    
    methods(Static)
        
        function result = mat2nparray( matarray )
            %mat2nparray Convert a Matlab array into an nparray
            %   Convert an n-dimensional Matlab array into an equivalent nparray
            data_size=size(matarray);
            if length(data_size)==1
                % 1-D vectors are trivial
                result=py.numpy.array(matarray);
            elseif length(data_size)==2
                % A transpose operation is required either in Matlab, or in Python due
                % to the difference between row major and column major ordering
                transpose=matarray';
                % Pass the array to Python as a vector, and then reshape to the correct
                % size
                result=py.numpy.reshape(transpose(:)', int32(data_size));
            else
                % For an n-dimensional array, transpose the first two dimensions to
                % sort the storage ordering issue
                transpose=permute(matarray, length(data_size):-1:1);
                % Pass it to python, and then reshape to the python style of matrix
                % sizing
                result=py.numpy.reshape(transpose(:)', int32(fliplr(size(transpose))));
            end
        end
        
        function result = nparray2mat( nparray )
            %nparray2mat Convert an nparray from numpy to a Matlab array
            %   Convert an n-dimensional nparray into an equivalent Matlab array
            data_size = cellfun(@int64,cell(nparray.shape));
            if prod(data_size) == 0
                % Zero-size matrix - still preserve shape
                result = zeros(data_size);
            elseif length(data_size)==1
                % This is a simple operation
                result=double(py.array.array('d', py.numpy.nditer(nparray)));
            elseif length(data_size)==2
                % order='F' is used to get data in column-major order (as in Fortran
                % 'F' and Matlab)
                result=reshape(double(py.array.array('d', ...
                    py.numpy.nditer(nparray, pyargs('order', 'F')))), ...
                    data_size);
            else
                % For multidimensional arrays more manipulation is required
                % First recover in python order (C contiguous order)
                result=double(py.array.array('d', ...
                    py.numpy.nditer(nparray, pyargs('order', 'C'))));
                % Switch the order of the dimensions (as Python views this in the
                % opposite order to Matlab) and reshape to the corresponding C-like
                % array
                result=reshape(result,fliplr(data_size));
                % Now transpose rows and columns of the 2D sub-arrays to arrive at the
                % correct Matlab structuring
                result=permute(result,[length(data_size):-1:1]);
            end
        end
        
        function test()
            A = 1:5;
            Anp = matpy.mat2nparray(A);
            
            sa = size(A);
            sAnp = cellfun( @(x) double(x), cell(Anp.shape));
            assert (all(sAnp == sa));
            for i1=1:size(A,1)
                for i2=1:size(A,2)
                    assert(A(i1,i2) == Anp.item(int32(i1-1), int32(i2-1)));
                end
            end
            Anpm = matpy.nparray2mat(Anp);
            assert(all(A == Anpm));
            
            A = reshape(1:6, [2,3]);
            Anp = matpy.mat2nparray(A);
            
            sa = size(A);
            sAnp = cellfun( @(x) double(x), cell(Anp.shape));
            assert (all(sAnp == sa));
            for i1=1:size(A,1)
                for i2=1:size(A,2)
                    assert(A(i1,i2) == Anp.item(int32(i1-1), int32(i2-1)));
                end
            end
            Anpm = matpy.nparray2mat(Anp);
            assert(all(all(A == Anpm)));
            
            A = reshape(1:(2*3*4), [2,3,4]);
            Anp = matpy.mat2nparray(A);
            
            sa = size(A);
            sAnp = cellfun( @(x) double(x), cell(Anp.shape));
            assert (all(sAnp == sa));
            for i1=1:size(A,1)
                for i2=1:size(A,2)
                    for i3=1:size(A,3)
                        display(sprintf('%d %d %d -> %f %f', i1,i2,i3, A(i1,i2,i3), Anp.item(int32(i1-1), int32(i2-1), int32(i3-1))));
                        assert (A(i1,i2,i3) == Anp.item(int32(i1-1), int32(i2-1), int32(i3-1)))
                    end
                end
            end
            Anpm = matpy.nparray2mat(Anp);
            assert(all(all(all(A == Anpm))));
        end
        
    end
    
end
