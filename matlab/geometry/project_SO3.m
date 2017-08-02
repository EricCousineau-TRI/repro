function [R] = project_SO3(M)
% See drake::math::ProjectMatToRotMat
assert(all(size(M) == [3, 3]));
de = det(M);
assert(abs(de) > eps);
[V, D] = eig(M' * M);
L = 1 ./ sqrt(diag(D));
if de < 0
    L(1) = -L(1);
end
R = M * V * diag(L) * V';
end
