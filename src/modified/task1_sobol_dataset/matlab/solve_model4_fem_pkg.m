function out = solve_model4_fem_pkg(params_in)
% 统一入口（可直接编译为 Python 包用）
% - 输入: JSON 字符串或 struct；可只给需要扫描的字段，其它走默认值
% - 输出: struct，含 ok/err_code/message/features/meta（以及需要的话 t/av）

  if nargin < 1
      params_in = struct();  % 允许无参调用，走默认参数
  end

  % ---------- 1) 解析输入 ----------
  if ischar(params_in) || isstring(params_in)
      P = jsondecode(params_in);
  elseif isstruct(params_in)
      P = params_in;
  else
      out = mk_error(100,'INPUT_TYPE','must be JSON string or struct'); return;
  end

  % ---------- 2) 默认名义参数（来自你 main） ----------
  D = struct( ...
    'mc',39000,'Jc',2e6,'l',1.548,'mm',1000, ...
    'k',4e7,'c',1e4, ...
    'i0',25,'h0',0.01,'kh',6500,'kv',40,'ka',0.2, ...
    'V',500/3.6, ...                         % m/s
    'L',25,'mg',6000,'EI',8.58e10,'kb',3.2e10,'kc',1e7, ...
    'omega1',9.30,'omega2',12.50,'zeta1',2.67/100,'zeta2',3.51/100, ...
    'rayleighM',[],'rayleighK',[], ...
    's',8,'e',50,'num_dt',100, ...
    'V_kmh',[] ...                           % 可传 km/h，自动覆盖 V
  );

  % 用输入覆盖默认
  fns = fieldnames(P);
  for i=1:numel(fns), D.(fns{i}) = P.(fns{i}); end
  if ~isempty(D.V_kmh), D.V = D.V_kmh/3.6; end

  % Rayleigh 系数缺省时按两模态阻尼推算
  if isempty(D.rayleighM) || isempty(D.rayleighK)
      RR = 2*[1/D.omega1, D.omega1; 1/D.omega2, D.omega2] \ [D.zeta1; D.zeta2];
      D.rayleighM = RR(1); D.rayleighK = RR(2);
  end

  % ---------- 3) 初始条件尺寸（随 s,e 自动匹配） ----------
  n  = D.s*D.e + 1;
  Nd = 2*n + D.s - 1;
  if ~isfield(D,'yv0') || isempty(D.yv0), D.yv0 = zeros(10,1); end
  if ~isfield(D,'vv0') || isempty(D.vv0), D.vv0 = zeros(10,1); end
  if ~isfield(D,'ub0') || isempty(D.ub0), D.ub0 = zeros(Nd,1); end
  if ~isfield(D,'vb0') || isempty(D.vb0), D.vb0 = zeros(Nd,1); end

  % ---------- 4) 必要的数值检查（可避免发散） ----------
  [ok,msg] = diag_guard(D);
  if ~ok
      out = mk_error(111,'UNSTABLE_GUESS',msg); return;
  end

  % ---------- 5) 调用原始仿真 ----------
  try
      [t,yv,vv,av] = solve_model4_fem( ...
        D.mc,D.Jc,D.l,D.mm,D.k,D.c,D.i0,D.h0,D.kh,D.kv,D.ka,D.V, ...
        D.L,D.mg,D.EI,D.kb,D.kc,D.rayleighM,D.rayleighK,D.s,D.e, ...
        D.yv0,D.vv0,D.ub0,D.vb0,D.num_dt);

      % ---------- 6) 轻量特征（与你 main 的 plot 一致：av(1,:)） ----------
      a_car = double(av(1,:));
      dt    = mean(diff(double(t)));
      feats = struct('a_car_peak', max(abs(a_car)), ...
                     'a_car_rms',  sqrt(mean(a_car.^2)), ...
                     'label_unstable', uint8(max(abs(a_car)) > 0.6)); % 阈值自定

      % 如需把时域也带回，按需开启（编译跨语言时不建议太大）：
      % raw = struct('t',t,'av1',a_car);

      out = struct('ok',true,'err_code',uint16(0),'message',"", ...
                   'features',feats, ...
                   'meta',struct('dt',dt,'n',uint32(numel(t)), ...
                                 'sim_version',"v1",'solver',"solve_model4_fem"));
  catch ME
      out = mk_error(500,'SIM_FAIL', getReport(ME,'basic','hyperlinks','off'));
  end
end

% —— 工具函数 —— %
function [ok,msg] = diag_guard(D)
  F0 = (D.mc+8*D.mm)*9.81/8;
  K0 = F0 / ( (D.i0/D.h0)^2 );
  Ci = 2*K0*D.i0/D.h0^2;
  Ch = 2*K0*D.i0^2/D.h0^3;
  diag_min = D.k + Ci*D.kh - Ch;
  ok = isfinite(diag_min) && (diag_min > 0);
  if ok, msg = ""; else, msg = sprintf('k + Ci*kh - Ch <= 0 (%.3e)',diag_min); end
end

function o = mk_error(code, tag, msg)
  o = struct('ok',false,'err_code',uint16(code), ...
             'message', string(tag) + ": " + string(msg));
end
