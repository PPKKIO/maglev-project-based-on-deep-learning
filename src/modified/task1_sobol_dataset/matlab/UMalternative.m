% Maglev Parameters
% Shanghai Transrapid

clear;clc;close all;

% Train Parameters
mc = 39000; % mass of car body
Jc = 2e6;   % mass inertia of car body
Lc = [10.836,7.740;4.644,1.548;-1.548,-4.644;-7.740,-10.836];
mb = 1500;  % mass of levitation bogie
Jb = 1200;  % mass inertia of levitation bogie
Lb = [1.548,-1.548;1.548,-1.548;1.548,-1.548;1.548,-1.548];
l = 1.548;
mm = 1000;  % mass of magnet
kp = 4e7;   % stiffness of primary suspension
cp = 1e4;   % damping of primary suspension
ks = 4e5;   % stiffness of secondary suspension
cs = 1e4;   % damping of secondary suspension
i0 = 25;    % nominal current
h0 = 0.01;  % nominal air gap
kh = 6500;  % feedback current gain (air gap)
kv = 40;    % feedback current gain (velocity)
ka = 0.2;   % feedback current gain (acceleration)

% Guideway Parameters
L  = 25;        % length of one span
mg = 6000;      % mass per unit length
EI = 8.58e10;   % flexural stiffness
kb = 3.2e10;    % stiffness of bearing support
kc = 1e7;       % stiffness of coupling connection

omega1 = 9.30;      % natural frequency of first mode
omega2 = 12.50;     % natural frequence of second mode
zeta1  = 2.67/100;  % damping ratio of first mode
zeta2  = 3.51/100;  % damping ratio of second mode
rayleigh = 2*[1/omega1,omega1;1/omega2,omega2]^-1*[zeta1;zeta2];
rayleighM = rayleigh(1); % Rayleigh damping coefficient (M)
rayleighK = rayleigh(2); % Rayleigh damping coefficient (K)

save('Maglev_Parameters.mat')

mv = mc+mb+mm;
V  = 500 / 3.6;  % m/s
s  = 8;
e  = 50;
s0 = 20;

yv0 = zeros(10,1);
vv0 = zeros(10,1);

n  = s*e+1;
Nd = 2 * n + s - 1;
ub0 = zeros(Nd,1);
vb0 = zeros(Nd,1);
k   = 4e7;
c   = 1e4;

% extra dimension
num_dt = 100;

[t,yv,vv,av] = solve_model4_fem(mc,Jc,l,mm,k,c,i0,h0,kh,kv,ka,V, ...
    L,mg,EI,kb,kc,rayleighM,rayleighK,s,e, ...
    yv0,vv0,ub0,vb0,num_dt);

% ---------- 后处理：整段 & 尾段最大加速度 ----------
a_car = av(1,:);   % 车体垂向加速度

% 整段峰值
a_peak_full = max(abs(a_car));

% 尾段比例（这里是最后 10%，想改就改这个数）
tail_frac = 0.10;
n_total   = numel(a_car);
idx_start = max(1, floor((1 - tail_frac) * n_total));
a_peak_tail = max(abs(a_car(idx_start:end)));

fprintf('整段峰值加速度: %g m/s^2\n', a_peak_full);
fprintf('最后 %.0f%% 段峰值加速度: %g m/s^2\n', tail_frac*100, a_peak_tail);

% 根据尾段峰值打标签举例（阈值你自己定）
THRESH = 0.6;  % 例子：0.6 m/s^2
label_unstable = a_peak_tail > THRESH;

% 可选：保存到文件，供后续统计 / 深度学习用
save('result_single_run.mat', ...
     't','a_car','a_peak_full','a_peak_tail','label_unstable');

% 画图
figure;
plot(t, a_car);
grid on;
xlabel('t (s)');
ylabel('a_{car} (m/s^2)');
title('Car body acceleration');
