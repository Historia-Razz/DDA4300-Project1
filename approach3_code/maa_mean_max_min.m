% 初始化变量
rng(33); % for reproducibility
num_runs = 10;
objective_values = zeros(num_runs, 1);
times = zeros(num_runs, 1);
iterations = zeros(num_runs, 1);

% approach2_import.m
% Load discrete distributions from test.d2 and compute Wasserstein barycenter
% 0. Parameters
dim = 2;        % Must match data file
options.support_size = 20;        % Desired number of barycenter support points
options.method = 'free_maaipm';
options.ipmouttolog = true;
options.largem = 0;
options.itmax = 2000;
options.ipmtol_primal_dual_gap = 1e-8;
support_size = options.support_size;
% 1. Read data from file
datafile = 'test_N15_mt100.d2';
fid = fopen(datafile, 'r');
if fid<0
    error('Cannot open file %s', datafile);
end

stride = [];
supp_all = [];
w_all = [];

while ~feof(fid)
    % Read dimension and number of points
    d = fscanf(fid, '%d', 1);
    mt = fscanf(fid, '%d', 1);
    
    % Read weights (1 x mt)
    w = fscanf(fid, '%f', [1, mt]);;
    
    % Read support points (d x mt)
    pts = fscanf(fid, '%f', [d, mt]);;
    
    % Collect
    stride = [stride, mt];
    w_all = [w_all, w];
    supp_all = [supp_all, pts];
    
    % Skip blank lines if any
    fgetl(fid);
    fgetl(fid);
end
fclose(fid);

% Construct db struct
db = cell(1,1);
db{1}.stride = stride;
db{1}.w      = w_all;
db{1}.supp   = supp_all;

for run = 1:num_runs
    % 2. Initial guess for barycenter c0
    tot_pts = sum(stride);
    % idx = randperm(tot_pts, support_size);
    % initial_support = supp_all(:, idx);
    % initial_weights = ones(support_size, 1)/support_size;
    initial_support = -5 + 10 * rand(2, support_size);
    initial_weights = rand(support_size, 1);
    initial_weights = initial_weights/sum(initial_weights);

    c0 = cell(1,1);
    c0{1}.supp = initial_support;
    c0{1}.w    = initial_weights;

    % 3. Solve Wasserstein barycenter
    tic;
    t0 = toc;
    [c, OT, iter_hist, optval_hist] = Wasserstein_Barycenter(db, c0, options);
    T = toc - t0;
    objective_values(run) = optval_hist(end);
    times(run) = T;
    iterations(run) = length(iter_hist);
end

% 打印平均值和最大最小值
fprintf('Average objective = %.4f\n', mean(objective_values));
fprintf('Average time = %.4f s\n', mean(times));
fprintf('Average iterations = %d\n', mean(iterations));
fprintf('Max objective = %.4f\n', max(objective_values));
fprintf('Min objective = %.4f\n', min(objective_values));
fprintf('Max time = %.4f s\n', max(times));
fprintf('Min time = %.4f s\n', min(times));
fprintf('Max iteration = %d\n', max(iterations));
fprintf('Min iterations = %d\n', min(iterations));