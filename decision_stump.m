function u = decision_stump(X,parameters)
    [N,D] = size(X);
    [n2,~] = size(parameters);
    X = reshape(X,[N,1,D]);
    parameters = reshape(parameters,[1,n2,D]);
    u = ones(N,n2,D);
    u(X < parameters) = -1;
end