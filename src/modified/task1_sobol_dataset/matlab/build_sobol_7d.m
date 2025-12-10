function build_sobol_7d(N, outPrefix)
% 生成 7 维 Sobol 参数，并保存为 .mat 和 .csv
% N: 样本数（建议 2^m，如 128, 256, 512, 1024）
% outPrefix: 输出文件前缀，如 'sobol7d_512'

    if nargin < 1, N = 1024; end
    if nargin < 2, outPrefix = sprintf('sobol7d_%d', N); end

    % ---------- 7 维参数设置 ----------
    names = {'k','c','h0','i0','kh','kv','V_kmh'};
    modes = {'log','log','lin','lin','lin','lin','lin'};

    lo = [ 3.0e7,  5.0e3,  8e-3, 20.0, 4.0e3, 20.0, 200.0];
    hi = [ 6.0e7,  2.0e4, 12e-3, 30.0, 9.0e3, 80.0, 500.0];

    d = numel(names);

    % ---------- 生成 Sobol 序列 U ∈ [0,1]^d ----------
    p = sobolset(d, 'Skip', 1e3, 'Leap', 1e2);      % 略过前面点，稳定些
    p = scramble(p, 'MatousekAffineOwen');          % 打乱，避免规则性

    U = net(p, N);   % N × d

    % ---------- 按 lin / log 映射到物理范围 ----------
    X = zeros(N, d);
    for j = 1:d
        if strcmpi(modes{j}, 'lin')
            X(:, j) = lo(j) + (hi(j) - lo(j)) .* U(:, j);
        else  % 'log'：几何插值
            X(:, j) = lo(j) .* (hi(j) / lo(j)) .^ U(:, j);
        end
    end

    % ---------- 保存为 table ----------
    T = array2table(X, 'VariableNames', names);

    % 保存 .mat
    matFile = [outPrefix, '.mat'];
    save(matFile, 'T');
    fprintf('保存 MAT: %s\n', matFile);

    % 保存 .csv（便于检查或给别的程序用）
    csvFile = [outPrefix, '.csv'];
    writetable(T, csvFile);
    fprintf('保存 CSV: %s\n', csvFile);
end
