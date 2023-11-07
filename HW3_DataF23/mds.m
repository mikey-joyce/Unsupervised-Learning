cities = load('MOCityDistF23.dat');

% Branson, Cape, Como, Jeff, KC, Rolla, STL, Spring, St. Joe
disp(cities);

% Perform classical multi-dimensional scaling
[coords, eigenvals] = cmdscale(cities, 2);

% Sort the eigenvalues
sorted_eigenvals = sort(eigenvals, 'descend');

% Find the second largest eigenvalue
largest_eigval = sorted_eigenvals(1);
second_largest_eigval = sorted_eigenvals(2);

disp("Largest eigenval: " + largest_eigval);
disp("Second largest eigenval: " + second_largest_eigval);

city_names = {'Branson','Cape Girardeau','Columbia','Jefferson City','Kansas City','Rolla','St. Louis','Springfield','St. Joseph'};

disp('Coordinates');
disp(coords);

disp('Euclidean distance between Como & St. Joe');
temp = [coords(3, :); coords(9,:)];
euc = pdist(temp);
disp(euc);

disp('Euclidean distance between KC & Springfield');
temp = [coords(5, :); coords(8,:)];
euc = pdist(temp);
disp(euc);

scatter(coords(:, 1), coords(:, 2), 'filled');
for i = 1:length(city_names)
    text(coords(i, 1), coords(i, 2), city_names{i});
end
title('MDS Plot');