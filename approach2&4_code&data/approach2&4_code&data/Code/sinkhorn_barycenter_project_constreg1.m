% ====== Sinkhorn-based Free Support Wasserstein Barycenter ======
% Full MATLAB Project
% Including: Simplex projection, Sinkhorn solver, Barycenter optimization
% Author: (based on user provided Python version)

%% Main script
% Generate synthetic distributions
% rng(115);
% Y_list = {};
% b_list = {};
% for i = 1:4
%     % 生成随机的均值和对称协方差矩阵
%     mean_vec = -5 + 10 * rand(1,2);
%     temp = rand(2) - 0.5;
%     cov_mat = randi([10,40]) * (eye(2) + 0.5*(temp + temp'));  % 保证对称
% 
%     Y = mvnrnd(mean_vec, cov_mat, randi([20,30]));
%     b = rand(size(Y,1),1);
%     b = b / sum(b);
%     Y_list{end+1} = Y;
%     b_list{end+1} = b;
% end

% sample2: deal with data from database
% Initialize the lists
Y_list = cell(db{1}.N, 1);
b_list = cell(db{1}.N, 1);
idx = 0
% Loop through each distribution
for i = 1:db{1}.N
    mi = db{1}.stride(i);
    points = db{1}.supp(:, idx+1: idx+mi)';
    weights = db{1}.w(idx+1: idx+mi)';

    % Reshape points and weights to the desired format
    Y_list{i} = reshape(points, mi, 2); % Reshape to (mi, 2) for x and y coordinates
    b_list{i} = weights; % Store weights directly
    % Update the index
    idx = idx + mi;
end

% Initialize support
num_support = 20;
X_init = -5 + 10 * rand(num_support, 2);
% a_init = ones(num_support, 1);
a_init = rand(num_support, 1);
a_init = a_init / sum(a_init);

% Solve barycenter
tic
[X_final, a_final, objective_values] = free_support_barycenter(Y_list, b_list, a_init, X_init, 1e-3, 0.5, 0.5, 0.7, 1e-10, 500, true);
toc
% Plotting - distributions and barycenter
figure;
hold on;
title('Wasserstein Barycenter of Random Bivariate Normal Distributions');
xlabel('X');
ylabel('Y');
grid on;

for i = 1:length(Y_list)
    Y = Y_list{i};
    b = b_list{i};
    scatter(Y(:,1), Y(:,2), b*1000, 'filled', 'MarkerFaceAlpha', 0.7);
end

scatter(X_final(:,1), X_final(:,2), a_final*100*num_support, a_final, 'filled', 'MarkerEdgeColor', 'k', 'Marker', 'O');
colorbar;
colormap('parula');
c = colorbar;
c.Label.String = 'Weights';
hold off;

% Plotting - objective convergence
figure;
plot(1:length(objective_values), objective_values, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('Iterations');
ylabel('Objective Value');
title('Objective Value Convergence');
grid on;

%% Function: project_simplex
function y = project_simplex(x)
    x(x < 0) = 0;
    if abs(sum(x)) < 1e-12
        y = zeros(size(x));
    else
        y = x / sum(x);
    end
end

%% Function: sinkhorn
function [T, loginfo] = sinkhorn(a, b, M, reg, numItermax, stopThr, warmstart, verbose)
    if nargin < 5, numItermax = 1000; end
    if nargin < 6, stopThr = 1e-9; end
    if nargin < 7, warmstart = []; end
    if nargin < 8, verbose = false; end

    [dim_a, dim_b] = size(M);
    Mr = -M / reg;

    if isempty(warmstart)
        u = zeros(dim_a, 1);
        v = zeros(dim_b, 1);
    else
        u = warmstart{1};
        v = warmstart{2};
    end

    loga = log(a);
    logb = log(b);

    for ii = 1:numItermax
        v = logb - log(sum(exp(Mr + u), 1))';
        u = loga - log(sum(exp(Mr + v'), 2));

        if mod(ii, 10) == 0
            tmp2 = sum(exp(Mr + u + v'), 2);
            err = norm(tmp2 - a);
            if verbose && mod(ii, 200) == 0
                fprintf('%5d | %e\n', ii, err);
            end
            if err < stopThr
                break;
            end
        end
    end

    logT = Mr + u + v';
    T = exp(logT);

    loginfo.log_u = u;
    loginfo.log_v = v;
    loginfo.u = exp(u);
    loginfo.v = exp(v);
end

%% Function: compute_alpha_star
function alpha = compute_alpha_star(u, v, reg)
    alpha = -reg * u;
    alpha = alpha + mean(reg * u);
end

%% Function: solve_barycenter
function [a, X_k, obj] = solve_barycenter(X_k, Y_list, a, B_list, M_list, reg, beta, theta, t_0)
    num_support = size(X_k, 1);
    Y_sum = zeros(size(X_k));
    alpha = zeros(num_support, 1);
    obj = 0;

    for i = 1:length(B_list)
        [T_i, res] = sinkhorn(a, B_list{i}, M_list{i}, reg);
        alpha = alpha + compute_alpha_star(res.log_u, res.log_v, reg);
        obj = obj + sum(sum(T_i .* M_list{i}));
        Y_sum = Y_sum + T_i * Y_list{i};
    end

    alpha = alpha / length(B_list);
    Y_sum = Y_sum / length(B_list);
    obj = obj / length(B_list);

    a_prev = a;
    a = a .* exp(-t_0 * alpha / beta);
    a = a / sum(a);
    a = (1 - beta) * a_prev + beta * a;

    X_k = (1 - theta) * X_k + theta * (diag(1./a) * Y_sum);
end

%% Function: free_support_barycenter
function [X, a, objective_values] = free_support_barycenter(Y_list, b_list, a_init, X_init, stopThr, reg, beta, theta, t_0, numItermax, verbose)
    if nargin < 5, stopThr = 1e-7; end
    if nargin < 6, reg = 0.1; end
    if nargin < 7, beta = 0.5; end
    if nargin < 8, theta = 0.5; end
    if nargin < 9, t_0 = 0.1; end
    if nargin < 10, numItermax = 100; end
    if nargin < 11, verbose = false; end

    X = X_init;
    a = a_init;
    obj = Inf;
    objective_values = [];

    for k = 1:numItermax
        X_prev = X;
        a_prev = a;
        obj_prev = obj;
        reg_k = 0

        M_list = cell(length(Y_list), 1);
        for i = 1:length(Y_list)
            Y = Y_list{i};
            M_list{i} = pdist2(X, Y).^2;
            medians = median(M_list{i}, 1);
            reg_k = reg_k + mean(medians);
        end

        reg_k = reg_k/(length(Y_list))/60;
        [a, X, obj] = solve_barycenter(X, Y_list, a, b_list, M_list, reg, beta, theta, t_0);

        dX = norm(X - X_prev, 'fro');
        da = norm(a - a_prev, 2);
        objective_gap = abs(obj - obj_prev) / (obj + 1e-9);
        objective_values(end+1) = obj;

        if verbose
            fprintf('[%d/%d] |dX|: %.3e, |da|: %.3e, obj: %.6f, gap: %.2e\n', k, numItermax, dX, da, obj, objective_gap);
        end

        if dX < 5e-2 && da < 1e-2 && objective_gap < 1e-2
            break;
        end
    end
end
