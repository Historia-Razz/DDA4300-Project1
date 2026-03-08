% generate_data.m
% Generate synthetic discrete distributions for Wasserstein barycenter experiments
% Each file contains N heterogeneous Gaussian distributions.

rng(0);  % For reproducibility

% Base parameters (length-7 lists)
mts   = [5, 10, 20, 40, 60, 80, 100];      % Base support sizes per file
Ns    = [3, 4, 6, 8, 10, 12, 15];          % Number of distributions per file
d     = 2;                                 % Dimensionality of support points

% Pools of possible means and variances (7 each)
mus   = { [0;0], [2;3], [5;5], [-7;2], [-4;-3], [-5;10], [10;-5] };
sig2s = { [1,1],  [3,3],  [5,5]};

% Loop over combinations of (mt_base, N)
for i_mt = 1:length(mts)
    mt_base = mts(i_mt);
    for i_N = 1:length(Ns)
        N = Ns(i_N);
        % File name encodes N and mt_base
        filename = sprintf('test_N%d_mt%d.d2', N, mt_base);
        fid = fopen(filename, 'wt');
        if fid < 0
            error('Cannot open file %s for writing.', filename);
        end
        
        % Generate N heterogeneous distributions
        for t = 1:N
            % Randomize support count around mt_base (50%–150%)
            mt = randi([floor(0.5*mt_base), ceil(1.5*mt_base)]);
            
            % 1) Header: dimension and support size
            fprintf(fid, '%d\n%d\n', d, mt);
            
            % 2) Weights: random +1e-3, normalized
            w = rand(1, mt) + 1e-3;
            w = w / sum(w);
            fprintf(fid, '%6.5f ', w);
            fprintf(fid, '\n');
            
            % 3) Randomly pick one (mu, sig2) for this distribution
            idx_mu    = randi(numel(mus));
            mu_t   = mus{idx_mu};
            idx_sig   = randi(numel(sig2s))
            sig2_t = sig2s{idx_sig};
            
            % 4) Support points: Gaussian with mean mu_t and var sig2_t
            pts = mvnrnd(mu_t', diag(sig2_t), mt)';  % d x mt
            for j = 1:mt
                fprintf(fid, '%8.6f ', pts(:, j));
                fprintf(fid, '\n');
            end
            fprintf(fid, '\n');
        end
        
        fclose(fid);
        fprintf('Wrote %s\n', filename);
    end
end
