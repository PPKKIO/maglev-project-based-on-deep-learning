% Maglev Model 
% 10-DOF Train on Flexible Guideway
% Finite Element Method

% Train Parameters
% mc = mass of car body
% Jc = mass moment of inertia of car body
% mb = mass of levitation bogie
% Jb = mass moment of inertia of levitation bogie
% mm = mass of magnet
% kp = stiffness of primary suspension
% cp = damping of primary suspension
% ks = stiffness of secondary suspension
% cs = damping of secondary suspension
% i0 = nominal current
% h0 = nominal air gap
% kh = feedback current gain (air gap)
% kv = feedback current gain (velocity)
% ka = feedback current gain (acceleration)

% yv0 = initial train displacement
% vv0 = initial train velocity

% Guideway Parameters
% L = length of one span
% mg = mass per unit length
% EI = flexural stiffness
% kb = stiffness of bearing support
% kc = stiffness of coupling connection
% rayleighM = Rayleigh damping coefficient (associated with mass matrix)
% rayleighK = Rayleigh damping coefficient (associated with stiffness matrix)

% s = number of spans
% e = number of elements per span

% ub0 = initial guideway displacement
% vb0 = initial guideway velocity

% Other Inputs
% num_dt = number of time steps per element

% Outputs
% t = time vector (for plotting of graphs)
% yv = train displacement
% vv = train velocity
% av = train acceleration

function [t,yv,vv,av] = solve_model4_fem(mc,Jc,l,mm,k,c,i0,h0,kh,kv,ka,V, ...
    L,mg,EI,kb,kc,rayleighM,rayleighK,s,e, ...
    yv0,vv0,ub0,vb0,num_dt)

% Train
F0 = (mc+8*mm)*9.81/8;
K0 = F0/(i0/h0)^2;
Ci = 2*K0*i0/h0^2;
Ch = 2*K0*i0^2/h0^3;

Lc = l*[7,5,3,1,-1,-3,-5,-7];

Mv = zeros(10,10);
Mv(1,1) = mc;
Mv(2,2) = Jc;
for i = 1:8
    Mv(i+2,i+2) = mm+Ci*ka;
end

Cv = zeros(10,10);
for i = 1:8
    Cv(1,1) = Cv(1,1)+c;
    Cv(1,i+2) = Cv(1,i+2)-c;
    Cv(2,2) = Cv(2,2)+c*Lc(i)^2;
    Cv(2,i+2) = Cv(2,i+2)-c*Lc(i);
    Cv(i+2,1) = Cv(i+2,1)-c;
    Cv(i+2,2) = Cv(i+2,2)-c*Lc(i);
    Cv(i+2,i+2) = Cv(i+2,i+2)+c+Ci*kv;
end

Kv = zeros(10,10);
for i = 1:8
    Kv(1,1) = Kv(1,1)+k;
    Kv(1,i+2) = Kv(1,i+2)-k;
    Kv(2,2) = Kv(2,2)+k*Lc(i)^2;
    Kv(2,i+2) = Kv(2,i+2)-k*Lc(i);
    Kv(i+2,1) = Kv(i+2,1)-k;
    Kv(i+2,2) = Kv(i+2,2)-k*Lc(i);
    Kv(i+2,i+2) = Kv(i+2,i+2)+k+Ci*kh-Ch;
end

% Guideway
n = s*e+1;
Le = L/e;

node_info = zeros(n,4);
node_info(:,1) = transpose(1:n);
node_info(1,2:4) = [2,1,0];
for i = 2:n-1
    if mod(i,e) == 1 || e == 1
        node_info(i,2:4) = [3,1,1];
    else
        node_info(i,2:4) = [2,0,0];
    end
end
node_info(end,2:4) = [2,1,0];

element_info = zeros(s*e,5);
element_info(:,1) = transpose(1:s*e);
element_info(1,2:5) = [1,2,3,4];
for i = 2:s*e
    element_info(i,2) = element_info(i-1,4);
    if node_info(i,4) == 1
        element_info(i,3) = element_info(i,2)+2;
    else
        element_info(i,3) = element_info(i,2)+1;
    end
    element_info(i,4) = element_info(i,3)+1;
    element_info(i,5) = element_info(i,4)+1;
end

Me = mg*Le/420*[156,22*Le,54,-13*Le;22*Le,4*Le^2,13*Le,-3*Le^2;54,13*Le,156,-22*Le;-13*Le,-3*Le^2,-22*Le,4*Le^2];
Ke = EI/Le^3*[12,6*Le,-12,6*Le;6*Le,4*Le^2,-6*Le,2*Le^2;-12,-6*Le,12,-6*Le;6*Le,2*Le^2,-6*Le,4*Le^2];
Ce = rayleighM*Me+rayleighK*Ke;

Mb = zeros(sum(node_info(:,2)),sum(node_info(:,2)));
Cb = zeros(sum(node_info(:,2)),sum(node_info(:,2)));
Kb = zeros(sum(node_info(:,2)),sum(node_info(:,2)));
for i = 1:s*e
    Mb(element_info(i,2:5),element_info(i,2:5)) = Mb(element_info(i,2:5),element_info(i,2:5))+Me;
    Cb(element_info(i,2:5),element_info(i,2:5)) = Cb(element_info(i,2:5),element_info(i,2:5))+Ce;
    Kb(element_info(i,2:5),element_info(i,2:5)) = Kb(element_info(i,2:5),element_info(i,2:5))+Ke;
end
Kb(1,1) = Kb(1,1)+kb;
for i = 2:n-1
    dof = sum(node_info(1:i-1,2))+1;
    if node_info(i,3) == 1
        Kb(dof,dof) = Kb(dof,dof)+2*kb;
    end
    if node_info(i,4) == 1
        Kb([dof+1,dof+2],[dof+1,dof+2]) = Kb([dof+1,dof+2],[dof+1,dof+2])+[kc,-kc;-kc,kc];
    end
end
Kb(end-1,end-1) = Kb(end-1,end-1)+kb;

% Numerical Integration
dt = L/V/e/num_dt;
t = dt*(0:s*e*num_dt);
yv = zeros(10,length(t));
vv = zeros(10,length(t));
av = zeros(10,length(t));
ub = zeros(2*n+s-1,length(t));
vb = zeros(2*n+s-1,length(t));
ab = zeros(2*n+s-1,length(t));

yv(:,1) = yv0;
vv(:,1) = vv0;
ub(:,1) = ub0;
vb(:,1) = vb0;

train_pos = l*[7,5,3,1,-1,-3,-5,-7];
Pv0 = zeros(10,1);
Pv0(1) = mc*9.81;
for j = 1:8
    if train_pos(j) >= 0 && train_pos(j) <= s*L
        pos = train_pos(j);

if pos <= 0
    element_cur = 1;
    x = 0;
elseif pos >= s*L
    element_cur = s*e;
    x = Le;   % 落在最后一个单元右端
else
    element_cur = ceil(pos/Le);       % 单元号，保证 ≥1
    x = pos - (element_cur-1)*Le;     % 单元内局部坐标
end

        N1 = 1-3*(x/Le)^2+2*(x/Le)^3;
        N2 = ((x/Le)-2*(x/Le)^2+(x/Le)^3)*Le;
        N3 = 3*(x/Le)^2-2*(x/Le)^3;
        N4 = (-(x/Le)^2+(x/Le)^3)*Le;
        dof1 = element_info(element_cur,2);
        dof2 = element_info(element_cur,3);
        dof3 = element_info(element_cur,4);
        dof4 = element_info(element_cur,5);
        u1 = ub(dof1,1);
        theta1 = ub(dof2,1);
        u2 = ub(dof3,1);
        theta2 = ub(dof4,1);
        u = [N1,N2,N3,N4]*[u1;theta1;u2;theta2];
    else
        u = 0;
    end
    Pv0(j+2) = mm*9.81-F0+(Ci*kh-Ch)*(u+h0);
end
av(:,1) = Mv\(Pv0-Cv*vv(:,1)-Kv*yv(:,1));
Pb0 = zeros(2*n+s-1,1);
for j = 1:8
    if train_pos(j) >= 0 && train_pos(j) <= s*L
        pos = train_pos(j);

if pos <= 0
    element_cur = 1;
    x = 0;
elseif pos >= s*L
    element_cur = s*e;
    x = Le;   % 落在最后一个单元右端
else
    element_cur = ceil(pos/Le);       % 单元号，保证 ≥1
    x = pos - (element_cur-1)*Le;     % 单元内局部坐标
end

        N1 = 1-3*(x/Le)^2+2*(x/Le)^3;
        N2 = ((x/Le)-2*(x/Le)^2+(x/Le)^3)*Le;
        N3 = 3*(x/Le)^2-2*(x/Le)^3;
        N4 = (-(x/Le)^2+(x/Le)^3)*Le;
        dof1 = element_info(element_cur,2);
        dof2 = element_info(element_cur,3);
        dof3 = element_info(element_cur,4);
        dof4 = element_info(element_cur,5);
        F = mm*9.81-mm*av(j+2,1)-c*(vv(j+2,1)-(vv(1,1)+Lc(j)*vv(2,1)))-k*(yv(j+2,1)-(yv(1,1)+Lc(j)*yv(2,1)));
        Pb0([dof1,dof2,dof3,dof4]) = Pb0([dof1,dof2,dof3,dof4])+F*[N1;N2;N3;N4];
    end
end
ab(:,1) = Mb\(Pb0-Cb*vb(:,1)-Kb*ub(:,1));

beta = 0.25;
gamma = 0.5;
a1 = 1/(beta*dt^2);
a2 = 1/(beta*dt);
a3 = 1/(2*beta)-1;
b1 = gamma/(beta*dt);
b2 = gamma/beta-1;
b3 = dt*(gamma/(2*beta)-1);

Kv_eff = Kv+a1*Mv+b1*Cv;
Kb_eff = Kb+a1*Mb+b1*Cb;

for i = 2:length(t)
    ub_prev = ub(:,i-1);

    err = Inf;
    while err > 1e-8
        ub_cur = ub_prev;

        vb_cur = b1*(ub_cur-ub(:,i-1))-b2*vb(:,i-1)-b3*ab(:,i-1);
        ab_cur = a1*(ub_cur-ub(:,i-1))-a2*vb(:,i-1)-a3*ab(:,i-1);

        train_pos = (i-1)*L/e/num_dt*ones(1,8)+l*[7,5,3,1,-1,-3,-5,-7];

        Pv = zeros(10,1);
        Pv(1) = mc*9.81;
        for j = 1:8
            if train_pos(j) >= 0 && train_pos(j) <= s*L
                pos = train_pos(j);

if pos <= 0
    element_cur = 1;
    x = 0;
elseif pos >= s*L
    element_cur = s*e;
    x = Le;   % 落在最后一个单元右端
else
    element_cur = ceil(pos/Le);       % 单元号，保证 ≥1
    x = pos - (element_cur-1)*Le;     % 单元内局部坐标
end

                N1 = 1-3*(x/Le)^2+2*(x/Le)^3;
                N2 = ((x/Le)-2*(x/Le)^2+(x/Le)^3)*Le;
                N3 = 3*(x/Le)^2-2*(x/Le)^3;
                N4 = (-(x/Le)^2+(x/Le)^3)*Le;
                dof1 = element_info(element_cur,2);
                dof2 = element_info(element_cur,3);
                dof3 = element_info(element_cur,4);
                dof4 = element_info(element_cur,5);
                u1 = ub_cur(dof1);
                theta1 = ub_cur(dof2);
                u2 = ub_cur(dof3);
                theta2 = ub_cur(dof4);
                u = [N1,N2,N3,N4]*[u1;theta1;u2;theta2];
            else
                u = 0;
            end
            Pv(j+2) = mm*9.81-F0+(Ci*kh-Ch)*(u+h0);
        end

        Pv_eff = Pv+(a1*Mv+b1*Cv)*yv(:,i-1)+(a2*Mv+b2*Cv)*vv(:,i-1)+(a3*Mv+b3*Cv)*av(:,i-1);
        yv_cur = Kv_eff\Pv_eff;

        vv_cur = b1*(yv_cur-yv(:,i-1))-b2*vv(:,i-1)-b3*av(:,i-1);
        av_cur = a1*(yv_cur-yv(:,i-1))-a2*vv(:,i-1)-a3*av(:,i-1);

        Pb = zeros(2*n+s-1,1);
        for j = 1:8
            if train_pos(j) >= 0 && train_pos(j) <= s*L
                pos = train_pos(j);

if pos <= 0
    element_cur = 1;
    x = 0;
elseif pos >= s*L
    element_cur = s*e;
    x = Le;   % 落在最后一个单元右端
else
    element_cur = ceil(pos/Le);       % 单元号，保证 ≥1
    x = pos - (element_cur-1)*Le;     % 单元内局部坐标
end

                N1 = 1-3*(x/Le)^2+2*(x/Le)^3;
                N2 = ((x/Le)-2*(x/Le)^2+(x/Le)^3)*Le;
                N3 = 3*(x/Le)^2-2*(x/Le)^3;
                N4 = (-(x/Le)^2+(x/Le)^3)*Le;
                dof1 = element_info(element_cur,2);
                dof2 = element_info(element_cur,3);
                dof3 = element_info(element_cur,4);
                dof4 = element_info(element_cur,5);
                F = mm*9.81-mm*av_cur(j+2)-c*(vv_cur(j+2)-(vv_cur(1)+Lc(j)*vv_cur(2)))-k*(yv_cur(j+2)-(yv_cur(1)+Lc(j)*yv_cur(2)));
                Pb([dof1,dof2,dof3,dof4]) = Pb([dof1,dof2,dof3,dof4])+F*[N1;N2;N3;N4];
            end
        end

        Pb_eff = Pb+(a1*Mb+b1*Cb)*ub(:,i-1)+(a2*Mb+b2*Cb)*vb(:,i-1)+(a3*Mb+b3*Cb)*ab(:,i-1);
        ub_cur = Kb_eff\Pb_eff;

        err = norm(ub_cur-ub_prev)/norm(ub_cur-ub(:,i-1));

        ub_prev = ub_cur;
    end

    yv(:,i) = yv_cur;
    vv(:,i) = vv_cur;
    av(:,i) = av_cur;
    ub(:,i) = ub_cur;
    vb(:,i) = vb_cur;
    ab(:,i) = ab_cur;
end
end