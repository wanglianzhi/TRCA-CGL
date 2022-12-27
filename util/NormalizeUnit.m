
function [ X ] = NormalizeUnit(X)

    [~,nSmp] = size(X);
    for j = 1:nSmp
        X(:, j) = ( X(:, j) - mean(X(:, j))) / std(X(:,j)); % mean 0 std 1
        X(:, j) = X(:, j) / norm(X(:, j)); % unit norm
    end
end