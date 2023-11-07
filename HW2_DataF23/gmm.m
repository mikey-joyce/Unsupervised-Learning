train = load('GMD_Train.dat');
validate = load('GMD_Valid.dat');
K = 3;
[train_mu, train_sigma, train_gamma, train_likelihoods, nk] = GMM_EM(train, K, 1);

disp('Mu');
disp(train_mu);

disp('Sigma');
disp(train_sigma);

disp('Nk');
disp(nk);

% Plot the data and GMM components
[~, clusters] = max(train_gamma, [], 2);

colors = [1, 0, 0;  % Red for color 1
                0, 0, 1;  % Blue
                0, 1, 0];
figure;
for i = 1:size(colors, 1)
    indices = (clusters == i);
    scatter(train(indices, 1), train(indices, 2), 50, colors(i, :), 'filled');
    hold on;
end

figure;
plot(1:20, train_likelihoods);
xlabel('iterations');
ylabel('Log-likelihood');

full_BIC = [];
spherical_BIC = [];
diagonal_BIC = [];
for i = 1:4
    [~, ~, ~, valid_likelihoods] = GMM_EM(validate, K, 1);
    full_BIC = [full_BIC, BIC(valid_likelihoods, validate, i)];

    [~, ~, ~, valid_likelihoods] = GMM_EM(validate, K, 2);
    spherical_BIC = [spherical_BIC, BIC(valid_likelihoods, validate, i)];

    [~, ~, ~, valid_likelihoods] = GMM_EM(validate, K, 3);
    diagonal_BIC = [diagonal_BIC, BIC(valid_likelihoods, validate, i)];
end

one = [full_BIC(1), spherical_BIC(1), diagonal_BIC(1)];
two = [full_BIC(2), spherical_BIC(2), diagonal_BIC(2)];
three = [full_BIC(3), spherical_BIC(3), diagonal_BIC(3)];
four = [full_BIC(4), spherical_BIC(4), diagonal_BIC(4)];

figure;
bar([one; two; three; four]);
legend({'Full', 'Spherical', 'Diagonal'});

function bic = BIC(likelihood, data, K)
    bic = -(-2*-likelihood(end) + log(K)*length(data));
end

% Gaussian Mixture Model EM Algorithm
function [mu, sigma, gamma, likelihoods, nk] = GMM_EM(data, K, which_model)
    max_iterations = 20;
    [samples, features] = size(data);  % get length of rows and columns

    mu = [5, 6; 10, 4; 2, -2; 0, 0];  % initialize the mean vectors
    mu = mu(1:K,:);    % only take the mean vectors we need
    %disp(mu);

    pi = ones(1, K)/K;
    sigma = repmat(eye(features), [1, 1, K]);

    likelihoods = [];
    for i = 1:max_iterations
        % evaluate responsibilities
        gamma = zeros(samples, K);

        for k = 1:K
            gamma(:, k) = pi(k) * mvnpdf(data, mu(k, :), sigma(:, :, k));
        end

        gamma_denom = sum(gamma, 2);
        gamma = gamma ./ gamma_denom;

        log_likelihood = sum(log(gamma_denom));
        likelihoods = [likelihoods, log_likelihood];

        % reestimate the parameters utilizing the responsiblities
        nk = sum(gamma, 1);
        pi = nk / samples;
        mu = (gamma' * data) ./ nk';
        for k = 1:K
            diff = data - mu(k, :);
            sigma(:, :, k) = (diff' * (diff .* gamma(:, k))) / nk(k);
            if which_model == 2
                % spherical
                sigma(2, 1, k) = 0;
                sigma(1, 2, k) = 0;
                sigma(1, 1, k) = (sigma(1, 1, k)+sigma(2,2,k))/2;
                sigma(2, 2, k) = sigma(1, 1, k);
            elseif which_model == 3
                % diagonal
                sigma(2, 1, k) = 0;
                sigma(1, 2, k) = 0;
            end
        end
    end
end