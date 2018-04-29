function [A, E] = ifralm(M)
shape = size(M);
fixed_rank = 2;
lam = 0.03;
magnitude = 1e-3;
batch_size = 10;
batch_num = ceil(shape(2) / batch_size);

%% fralm for initial batch
if batch_num > 1
    D = M(:, 1 : batch_size);
else
    D = M;
end
shp = size(D);
sgn = sign(D);
scl = max(norm(D, 2), norm(D, inf) / lam);
	
A = zeros(shp);
E = zeros(shp);
Y = sgn / scl;
A_last = A;
E_last = E;

s = svd(Y, 'econ');
mu = 0.5 / s(1);
pho = 2;

in_stop = magnitude * norm(D, 'fro');
out_stop = in_stop / 10;

in_converge = false;
out_converge = false;
while ~out_converge
    while ~in_converge
        disp('We are here 1')
        [U, S, V] = svd(D - E + Y / mu, 'econ');
        S = soft_threshhold(S, 1 / mu);
        if sum(diag(S)) < fixed_rank 
            A = (U .* diag(S)') * V';
        else
            vec = diag(S)';
            vec(fixed_rank + 1 : end) = 0;
            A = (U .* vec) * V';
        end
        E = soft_threshhold(D - A + Y / mu, lam / mu);
        in_converge = norm(A - A_last, 'fro') < in_stop && norm(E - E_last, 'fro') < in_stop;
        disp(strcat('A convergence :', num2str(norm(A - A_last, 'fro'))))
        disp(strcat('E convergence :', num2str(norm(E - E_last, 'fro'))))
        A_last = A;
        E_last = E;
    end
    in_converge = false;
    out_converge = norm(D - A - E, 'fro') < out_stop;
    disp(strcat('D convergence :', num2str(norm(D - A - E, 'fro'))))
    Y = Y + mu * (D - A - E);
    mu = pho * mu;
end
if batch_num == 1
    return
end
%% iFrALM
U_lastbatch = U(:, 1 : fixed_rank);
S_lastbatch = S(1 : 1 : fixed_rank, 1 : 1 : fixed_rank);
V_lastbatch = V(:, 1 : 1 : fixed_rank);

batch_sumlen = batch_size;
batch_order = 2;
disp('we are here 2')
for i = 2 : batch_num
    if batch_sumlen + batch_size > shape(2)
        batch_len = shape(2) - batch_sumlen;
    else
        batch_len = batch_size;
    end
    D = M(:, batch_sumlen + 1 : batch_sumlen + batch_len);
    sgn = sign(D);
    scl = max(norm(D, 2), norm(D, inf) / lam);

    A = zeros(size(D));
    E = zeros(size(D));
    Y = sgn / scl;
    A_lastiter = A;
    E_lastiter = E;
    
    s = svd(Y, 'econ');
    mu = 0.5 / s(1);
    pho = 6;
    
    in_stop = magnitude * norm(D, 'fro');
    out_stop = in_stop / 10;

    in_converge = false;
    out_converge = false;
    
    while ~out_converge
        while ~in_converge
            [U, S, V] = isvd(U_lastbatch, S_lastbatch, V_lastbatch, D - E + Y / mu);
            A = (U .* diag(S)') * V(batch_sumlen + 1 : batch_sumlen + batch_len, :)';
            E = soft_threshhold(D - A + Y / mu, lam / mu);
            in_converge = norm(A - A_lastiter, 'fro') < in_stop && norm(E - E_lastiter, 'fro') < in_stop;
            A_lastiter = A;
            E_lastiter = E;
            disp('we are here')
        end
        in_converge = false;
        out_error = norm(D - A - E, 'fro');
        out_converge =  out_error < out_stop;
        [U_tilde, S_tilde, V_tilde] = svd(S * V(1 : batch_sumlen, :)');
        U_lastbatch = U * U_tilde;
        S_lastbatch = S_tilde;
        V_lastbatch = V_tilde;
        Y = Y + mu * (D - A - E);
        mu = pho * mu;
    end
    U_lastbatch = U;
    S_lastbatch = S;
    V_lastbatch = V;
    batch_sumlen = batch_sumlen + batch_len;
    batch_order = batch_order + 1;
end



end



%% soft threshhold
function S =  soft_threshhold(S, tau)
     S = sign(S) .* max(abs(S) - tau, 0);
end

%%iSVD
function [U, S, V] = isvd(P, S, Q, D)
    % Ak =  P * S * Q'
    r = size(S, 1);
    [P_tilde, R] = qr((1 - P * P') * D, 0);
    
    B = zeros(size(S) + size(R));
    B(1 : r, 1 : r) = S;
    B(1 : r, r + 1 : end) = P' * D;
    B(1 : size(R, 1), r + 1: end) = R;
    
    [U_tilde, S_tilde, V_tilde] = svd(B, 'econ');
    U_tilde = U_tilde(:, 1 : r);
    S_tilde = S_tilde(1 : r, 1 : r);
    V_tilde = V_tilde(:, 1 : r);
    U = zeros(size(P, 1), size(P, 2) + size(P_tilde, 2));
    U(:, 1 : r) = P;
    U(:, r + 1 : end) = P_tilde;
    U = U * U_tilde;
    S = S_tilde;
    V = zeros(size(Q) + size(R));
    V(1 : size(Q, 1), 1 : size(Q, 2)) = Q;
    V(size(Q, 1) + 1 : end, size(Q, 2) + 1 : end) = eye(size(R, 1));
    V = V * V_tilde;
end
