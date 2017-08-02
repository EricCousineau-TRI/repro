% See: https://github.com/robotlocomotion/drake/pull/6752
M = reshape(1:9, [3, 3])';
M(3, 3) = 10;

det(M)
R = project_O3(M)
R2 = project_SO3(M)

M(1, 1) = 20;  % Augment this to get det(M) > 0
% We get about the same thing
project_O3(M) - project_SO3(M)
