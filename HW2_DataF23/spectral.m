halfmoon = load('HalfMoonF23.txt');

figure;
scatter(halfmoon(:, 1), halfmoon(:, 2), 'r', 'filled');

sigma = 1;
epsilon = 0.5;

distances = pdist2(halfmoon, halfmoon);
affinity = exp(-distances.^2 / (2 * sigma^2));
affinity(distances < epsilon) = 0;

laplacian = diag(sum(affinity, 2)) - affinity;
[eigen_vecs, eigen_vals] = eig(laplacian);

eigen_vals = diag(eigen_vals);

figure;
scatter(1:length(eigen_vals), eigen_vals, 'filled');
title('Eigenvalues');

sorted_vals = sort(eigen_vals);
second_smallest = min(sorted_vals(sorted_vals > sorted_vals(1)));
index_second = find(eigen_vals == second_smallest);

vec = eigen_vecs(index_second, :);

figure;
plot(1:length(vec), vec, 'b', 'LineWidth', 2);
title('Eigenvector belonging to 2nd smallest eigen value');
%stem(1:length(vec), vec, 'b', 'filled');

thresh = mean(vec);

clusters = [];
for i = 1:length(vec)
    if vec(i) >= thresh
        clusters = [clusters, 1];
    elseif vec(i) < thresh
        clusters = [clusters, 2];
    end
end

figure;
scatter(halfmoon(:, 1), halfmoon(:, 2), 50, clusters, 'filled');
title('Clustering Results');
