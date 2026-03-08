function [c_bar, iter_hist, obj_hist] = vanilla_ipm_wdp(db, c0, options)
% vanilla_ipm_wdp Solve Wasserstein barycenter via primal barrier IPM
%
% db: cell array with db{1}.stride, db{1}.supp, db{1}.w
% c0: initial barycenter struct with .supp (d x m) and .w (1 x m)
% options: struct with fields:
%   .maxBarrierIter (default 5)
%   .maxNewtonIter  (default 20)
%   .tolNewton      (default 1e-6)
%   .mu             (barrier parameter increase, default 10)

% Unpack db
stride = db{1}.stride;
supp    = db{1}.supp;
w       = db{1}.w;
N = length(stride);
d = size(supp,1);
m = length(c0.w);

% Build A and b and cost c
M = length(w);
n_row = M + N*(m-1) + 1;
n_col = M*m + m;

% Row sums for source marginals
row1 = kron((1:M)', ones(m,1));
% Row sums for barycenter marginals
row2 = zeros(M*(m-1),1);
cum = [0,cumsum(stride)];
for i=1:N
    idx = (cum(i)*m - (i-1)) + (1:(stride(i)*(m-1)));
    row2(idx) = kron(ones(stride(i),1), M + (i-1)*(m-1) + (1:(m-1))');
end
row3 = n_row * ones(m,1);
rows = [row1; row2; row3];

% Col indices
col1 = reshape(repmat(1:M, m,1),[],1);
col2 = repmat((M*(1:N) + (1:(m-1)))', stride,1);
col3 = M*m + (1:m)';
cols = [col1; col2; col3];

% Values
vals = ones(length(rows),1);

% Assemble A and b
A = sparse(rows, cols, vals, n_row, n_col);
b = zeros(n_row,1);
b(1:M) = w';
b(end) = 1;

% Cost vector
Cmat = pdist2(c0.supp', supp','sqeuclidean');
cost = [reshape(Cmat,[],1); zeros(m,1)];

% Initialize x>0 strictly feasible: use positive c0.w
% For transports, split w evenly across m supports
X0 = repmat(w'/m, m,1);
x = [X0(:); c0.w(:)];

% Barrier parameters
t = 1; mu = getfield(options,'mu',10);
maxBarrier = getfield(options,'maxBarrierIter',5);
maxNewton  = getfield(options,'maxNewtonIter',20);
tolNewton  = getfield(options,'tolNewton',1e-6);

iter_hist = [];
obj_hist  = [];

for barrierIter=1:maxBarrier
    % Centering step via Newton
    for newtonIter=1:maxNewton
        % Gradient and Hessian of barrier problem
        grad = t*cost - 1./x;
        H = spdiags(1./x.^2, 0, n_col, n_col);
        % KKT system: [H A'; A 0]*[dx; y] = -[grad; A*x - b]
        KKT = [H, A'; A, sparse(n_row,n_row)];
        rhs = -[grad; A*x - b];
        sol = KKT \ rhs;
        dx  = sol(1:n_col);
        % line search
        alpha = 1;
        idx = dx<0;
        if any(idx)
            alpha = min(1,0.99*min(-x(idx)./dx(idx)));
        end
        % Update
        x = x + alpha*dx;
        % check Newton decrement
        dec = sqrt(dx'*(H*dx));
        if dec < tolNewton
            break;
        end
    end
    % Record
    iter_hist(end+1) = barrierIter;
    obj_hist(end+1)  = cost'*x;
    % Increase t
    t = mu * t;
end

% Extract barycenter
theta = x(end-m+1:end);
c_bar.supp = c0.supp;
c_bar.w    = theta / sum(theta);

end
