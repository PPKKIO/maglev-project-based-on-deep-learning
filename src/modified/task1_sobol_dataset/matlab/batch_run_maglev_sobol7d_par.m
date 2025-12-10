function batch_run_maglev_sobol7d_par()
% 并行批量运行 7 维 Sobol 参数，对应 solve_model4_fem
% 要求先有 sobol7d_1024.mat，里边是 table T（k,c,h0,i0,kh,kv,V_kmh）

clear; clc;

%% 1. 读取 Sobol 参数表
paramFile = 'sobol7d_1024.mat';  % 你生成的 Sobol 文件名
S = load(paramFile);             % S.T 是 table
T = S.T;
N = height(T);

fprintf('读取 Sobol 参数：%s ，共 %d 组样本。\n', paramFile, N);

%% 2. 固定参数（不随 Sobol 变化）
% Train Parameters（除 7 维外其余保持名义值）
mc = 39000;
Jc = 2e6;
l  = 1.548;
mm = 1000;
ka = 0.2;

% Guideway Parameters
L  = 25;
mg = 6000;
EI = 8.58e10;
kb = 3.2e10;
kc = 1e7;

omega1 = 9.30;
omega2 = 12.50;
zeta1  = 2.67/100;
zeta2  = 3.51/100;
rayleigh = 2*[1/omega1,omega1;1/omega2,omega2]^-1*[zeta1;zeta2];
rayleighM = rayleigh(1);
rayleighK = rayleigh(2);

s  = 8;
e  = 50;
num_dt = 100;

% 初始条件
yv0 = zeros(10,1);
vv0 = zeros(10,1);
n   = s*e+1;
Nd  = 2*n + s - 1;
ub0 = zeros(Nd,1);
vb0 = zeros(Nd,1);

%% 3. 结果预分配
a_peak_full     = nan(N,1);
a_peak_tail     = nan(N,1);
label_unstable  = nan(N,1);
ok_flag         = false(N,1);

tail_frac = 0.10;     % 最后 10%
THRESH    = 0.6;      % 稳定性阈值，可自己改

checkpointFile = 'batch_results_checkpoint_par.mat';

%% 4. 并行池设置（自动按 CPU 配置）
try
    p = gcp('nocreate');
    if isempty(p)
        numCores = feature('numcores');    % 逻辑核心数
        % 保守一点：核数 *0.75，最多 8 个 worker
        numWorkers = min(8, max(2, floor(0.75 * numCores)));
        fprintf('准备开启并行池：%d workers（CPU 核心数 = %d）。\n', numWorkers, numCores);
        parpool('local', numWorkers);
    else
        fprintf('复用已存在的并行池：%d workers。\n', p.NumWorkers);
    end
catch ME
    warning('无法开启并行池，将退回串行 for：%s', ME.message);
end

%% 5. 按批次并行（每批 10 组）
batchSize = 10;

for batchStart = 1:batchSize:N
    batchEnd = min(batchStart + batchSize - 1, N);
    idxBatch = batchStart:batchEnd;
    nb = numel(idxBatch);

    fprintf('=== 正在并行计算样本 %d ~ %d / %d ===\n', batchStart, batchEnd, N);

    % parfor 直接写入预分配数组（按 loop 变量 i 切片）
    parfor i = idxBatch
        % 取第 i 行 Sobol 参数
        k   = T.k(i);
        c   = T.c(i);
        h0  = T.h0(i);
        i0  = T.i0(i);
        kh  = T.kh(i);
        kv  = T.kv(i);
        V   = T.V_kmh(i) / 3.6;   % km/h -> m/s

        try
            [t,yv,vv,av] = solve_model4_fem(mc,Jc,l,mm,k,c,i0,h0,kh,kv,ka,V, ...
                L,mg,EI,kb,kc,rayleighM,rayleighK,s,e, ...
                yv0,vv0,ub0,vb0,num_dt);

            a_car = av(1,:);

            % 整段峰值
            a_peak_full(i) = max(abs(a_car));

            % 尾段峰值
            n_total   = numel(a_car);
            idx_start = max(1, floor((1 - tail_frac) * n_total));
            a_peak_tail(i) = max(abs(a_car(idx_start:end)));

            % 稳定性标签
            label_unstable(i) = a_peak_tail(i) > THRESH;

            ok_flag(i) = true;
        catch
            ok_flag(i)         = false;
            a_peak_full(i)     = NaN;
            a_peak_tail(i)     = NaN;
            label_unstable(i)  = 1;   % 失败一律当不稳定，你可以改策略
        end
    end

    % ---- 每批结束后保存一次进度 ----
    resultsTable = table((1:N)', ok_flag, ...
        a_peak_full, a_peak_tail, label_unstable, ...
        'VariableNames', {'idx','ok','a_peak_full','a_peak_tail','label_unstable'});

    save(checkpointFile, 'T', 'resultsTable', 'tail_frac', 'THRESH');
    fprintf('>>> 已保存进度到 %s （已完成到 %d / %d）\n', checkpointFile, batchEnd, N);
end

%% 6. 最终导出 CSV 汇总
finalCsv = 'sobol7d_results_par.csv';
resultsTable = table((1:N)', ok_flag, ...
    a_peak_full, a_peak_tail, label_unstable, ...
    'VariableNames', {'idx','ok','a_peak_full','a_peak_tail','label_unstable'});

writetable(resultsTable, finalCsv);
fprintf('### 全部并行仿真结束，结果已写入 %s ###\n', finalCsv);

end
