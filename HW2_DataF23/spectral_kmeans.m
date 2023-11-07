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


twosq_twocirc = load('TwoSquaresTwoCircles.dat');

figure;
scatter(twosq_twocirc(:, 1), twosq_twocirc(:, 2), 'r', 'filled');

sigma = 0.5;
epsilon = 1;

distances = pdist2(twosq_twocirc, twosq_twocirc);
affinity = exp(-distances.^2 / (2 * sigma^2));
affinity(distances < epsilon) = 0;

laplacian = diag(sum(affinity, 2)) - affinity;
[eigen_vecs, eigen_vals] = eig(laplacian);
eigen_vals = diag(eigen_vals);

figure;
scatter(1:length(eigen_vals), eigen_vals, 'filled');
title('Eigenvalues');

sorted_vals = sort(eigen_vals);
counter=1;
while sorted_vals(counter) <= 0
    counter = counter + 1;
end
second_smallest = sorted_vals(counter);
third_smallest = sorted_vals(counter+1);
fourth_smallest = sorted_vals(counter+2);

for i=1:1200
    if eigen_vals(i) == second_smallest
        index_second = i;
    end
    if eigen_vals(i) == third_smallest
        index_third = i;
    end
    if eigen_vals(i) == fourth_smallest
        index_fourth = i;
    end
end

second = eigen_vecs(index_second, :);
third = eigen_vecs(index_third, :);
fourth = eigen_vecs(index_fourth, :);

eigen_vecs = horzcat(second', third', fourth');

figure;
plot(1:length(second), second, 'b', 'LineWidth', 2);
hold on;
plot(1:length(third), third, 'r', 'LineWidth', 2);
hold on;
plot(1:length(fourth), fourth, 'g', 'LineWidth', 2);
title('2nd, 3rd, and 4th Eigenvectors');

num_clusters = 4;
max_iter = 100;
centroid_1 = eigen_vecs(150, :);
centroid_2 = eigen_vecs(450, :);
centroid_3 = eigen_vecs(800, :);
centroid_4 = eigen_vecs(1100, :);

centroids = [centroid_1; centroid_2; centroid_3; centroid_4];

% k means clustering
clusters = zeros(size(eigen_vecs, 1), 1);
for i = 1:max_iter
    for j = 1:size(eigen_vecs, 1)
        distances = sum((centroids - eigen_vecs(i, :)).^2, 2);
        [~, clusters(i)] = min(distances);
    end

    for k = 1:num_clusters
        points_in_cluster = eigen_vecs(clusters == k, :);
        if ~isempty(points_in_cluster)
            centroids(k, :) = mean(points_in_cluster, 1);
        end
    end
end

figure;
scatter(twosq_twocirc(:, 1), twosq_twocirc(:, 2), 50, clusters, 'filled');
colormap(parula(num_clusters));

figure;
scatter(twosq_twocirc(:, 1), twosq_twocirc(:, 2), 50, clusters, 'filled');
colormap(parula(num_clusters));
hold on;

for k = 1:num_clusters
    scatter(centroids(k, 1), centroids(k, 3), 100, 'x', 'LineWidth', 2, 'DisplayName', ['Centroid ' num2str(k)]);
end
title('K-means Spectral Clustering Results');