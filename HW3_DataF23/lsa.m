W = load('WordDocF23.dat');

% SVD decomposition

[U, S, V] = svd(W, 'econ');
disp("Largest Singular Value: " + S(1));

% eigen vals
cov_matrix = cov(W);
eigenvals = eig(cov_matrix);
sorted_eigenvals = sort(eigenvals, 'descend');
largest_eigval = sorted_eigenvals(1);
disp("Largest eigenval: " + largest_eigval);

% Rank k approximation

k = 1;
Wk = U(:, 1:k) * S(1:k, 1:k) * V(:, 1:k)';
error = norm(W - Wk, 'fro');
disp("k=1: " + error);

k = 5;
Wk = U(:, 1:k) * S(1:k, 1:k) * V(:, 1:k)';
error = norm(W - Wk, 'fro');
disp("k=5: " + error);

k = 3;
U = U(:, 1:k);
S = S(1:k, 1:k);
V =  V(:, 1:k)';
Wk = U*S*V;
error = norm(W - Wk, 'fro');
disp("k=3: " + error);

% Inner product similarity

numRows = size(Wk, 1);
maxSimilarity = 0;
mostSimilarPair = [0, 0];

for i = 1:numRows
    for j = i+1:numRows
        % Calculate the inner product (dot product) between rows i and j
        similarity = dot(Wk(i, :), Wk(j, :));
        
        % Update if a higher similarity is found
        if similarity > maxSimilarity
            maxSimilarity = similarity;
            mostSimilarPair = [i, j];
        end
    end
end

disp("Inner Product Similarity k=3");
disp("Document Pair: ");
disp(mostSimilarPair);
disp("Similarity Value: " + maxSimilarity);

numCols = size(Wk, 2);
maxSimilarity = 0;
mostSimilarPair = [0, 0];

for i = 1:numCols
    for j = i+1:numCols
        % Calculate the inner product (dot product) between rows i and j
        similarity = dot(Wk(:, i), Wk(:, j));
        
        % Update if a higher similarity is found
        if similarity > maxSimilarity
            maxSimilarity = similarity;
            mostSimilarPair = [i, j];
        end
    end
end

disp("Word Pair: ");
disp(mostSimilarPair);
disp("Similarity Value: " + maxSimilarity);

% Cosine Similarity now

numRows = size(Wk, 1);
maxSimilarity = 0;
mostSimilarPair = [0, 0];

for i = 1:numRows
    for j = i+1:numRows
        % Calculate the inner product (dot product) between rows i and j
        row1 = Wk(i, :);
        row2 = Wk(j, :);
        similarity = dot(row1, row2) / (norm(row1) * norm(row2));
        
        % Update if a higher similarity is found
        if similarity > maxSimilarity
            maxSimilarity = similarity;
            mostSimilarPair = [i, j];
        end
    end
end

disp("Cosine Similarity k=3");
disp("Document Pair: ");
disp(mostSimilarPair);
disp("Similarity Value: " + maxSimilarity);

numCols = size(Wk, 2);
maxSimilarity = 0;
mostSimilarPair = [0, 0];

for i = 1:numCols
    for j = i+1:numCols
        % Calculate the inner product (dot product) between rows i and j
        col1 = Wk(:, i);
        col2 = Wk(:, j);
        similarity = dot(col1, col2) / (norm(col1) * norm(col2));
        
        % Update if a higher similarity is found
        if similarity > maxSimilarity
            maxSimilarity = similarity;
            mostSimilarPair = [i, j];
        end
    end
end

disp("Word Pair: ");
disp(mostSimilarPair);
disp("Similarity Value: " + maxSimilarity);
