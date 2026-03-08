% approach2_import.m
% Load discrete distributions from test.d2 and compute Wasserstein barycenter
rng(3); % for reproducibility
% 0. Parameters
dim = 2;        % Must match data file
options.support_size = 5;        % Desired number of barycenter support points
options.method = 'fixed_maaipm';
options.ipmouttolog = true;
options.largem = 0;
options.itmax = 2000;
options.ipmtol_primal_dual_gap = 1e-8;
support_size = options.support_size;
% 1. Read data from file
datafile = 'test_N6_mt100.d2';
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

% 2. Initial guess for barycenter c0
tot_pts = sum(stride);
idx = randperm(tot_pts, support_size);
initial_support = supp_all(:, idx);
initial_weights = ones(support_size, 1)/support_size;

c0 = cell(1,1);
c0{1}.supp = initial_support;
c0{1}.w    = initial_weights;

% 3. Solve Wasserstein barycenter
[c, OT] = Wasserstein_Barycenter(db, c0, options);

% 4. Plot results with weighted scatter
figure;
hold on;
% build cell lists of original supports and weights
offset = 0;
Y_list = cell(1,length(stride));
b_list = cell(1,length(stride));
for i=1:length(stride)
    mt_i = stride(i);
    Y_list{i} = supp_all(:,offset+1:offset+mt_i);
    b_list{i} = w_all(offset+1:offset+mt_i);
    offset = offset + mt_i;
end
% scatter each original distribution
for i=1:length(Y_list)
    Y = Y_list{i};
    b = b_list{i};
    scatter(Y(1,:),Y(2,:), b*1000, 'filled', 'MarkerFaceAlpha', 0.7);
end
% scatter barycenter supports
X_final = c{1}.supp;
a_final = c{1}.w;
num_s = size(X_final,2);
scatter(X_final(1,:), X_final(2,:), a_final*100*num_s, a_final, 'filled', 'MarkerEdgeColor', 'k', 'Marker', 'o');
colormap('parula');
c = colorbar;
c.Label.String = 'weights';
hold off;


