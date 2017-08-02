function [R] = project_O3(M)
% See drake::math::ProjectMatToOrthornomalMat
assert(all(size(M) == [3, 3]));

[U, ~, V] = svd(M);
R = U * V';

end
