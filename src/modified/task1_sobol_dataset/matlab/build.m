% build.m —— 兼容旧版SDK的打包脚本（不依赖 res.Summary）
cd(fileparts(mfilename('fullpath')));   % 切到 matlab/ 目录

entry  = "solve_model4_fem_pkg.m";      % 入口函数
pkg    = "maglev_sim";                  % Python 包名
verstr = "1.0.0";                       % 版本号（旧版可能不生效）
outDir = fullfile(pwd, "dist_py");      % 自定义输出目录（推荐）

if ~exist(outDir, 'dir'); mkdir(outDir); end

if exist('compiler.build.PythonPackageOptions','class')
    % 旧版/某些版本需要 Options 对象
    opts = compiler.build.PythonPackageOptions(entry);
    opts.PackageName = pkg;
    try
        opts.Version = verstr;          % 旧版不支持就忽略
    catch
        warning('当前版本不支持设置 Version，使用默认版本号。');
    end
    try
        opts.OutputDir = outDir;        % 把产物放到我们指定的目录
    catch
        % 某些版本没有 OutputDir 字段；忽略，让它用默认，再从默认处拷贝
        warning('当前版本不支持 OutputDir 字段，将从默认目录中查找 .whl。');
    end
    compiler.build.pythonPackage(opts);
else
    % 新一点的版本常见 Name-Value 形式；不传 Version 以避免不兼容
    try
        compiler.build.pythonPackage(entry, ...
            "PackageName", pkg, ...
            "OutputDir",   outDir, ...
            "Verbose",     "on");
    catch
        % 若这版不支持 OutputDir，则退回最朴素的调用
        compiler.build.pythonPackage(entry, ...
            "PackageName", pkg, ...
            "Verbose",     "on");
    end
end

% —— 查找 .whl 的位置（无论输出到哪，最终都能找到）——
candidates = [ ...
    dir(fullfile(outDir, '**', '*.whl')); ...
    dir(fullfile(pwd,   '**', '*.whl')) ...
];
if isempty(candidates)
    warning('未在 %s 或其子目录发现 .whl，请打开 Library Compiler 的输出目录查看。', outDir);
else
    fprintf('找到以下 wheel 文件：\n');
    for k = 1:numel(candidates)
        fprintf('  %s\n', fullfile(candidates(k).folder, candidates(k).name));
    end
end
