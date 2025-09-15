% Maglev Model 2
% 10-DOF Train on Flexible Guideway
% Moving Element Method

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

% s = number of spans (travelled by the train)
% e = number of elements per span

% ub0 = initial guideway displacement
% vb0 = initial guideway velocity

% Other Inputs
% s0 = number of spans (in the truncated guideway)

% Outputs
% t = time vector (for plotting of graphs)
% yv = train displacement
% vv = train velocity
% av = train acceleration

function [t,yv,vv,av] = solve_model4_mem(mc,Jc,l,mm,k,c,i0,h0,kh,kv,ka,V, ...
    L,mg,EI,kb,kc,rayleighM,rayleighK,s,e, ...
    yv0,vv0,~,~,s0)
% addition

n = s0 * e + 1;

% 构造 node_info，计算 DOF 总数
node_info = zeros(n,4);
node_info(:,1) = (1:n)';
node_info(1,2:4) = [2,1,1];
for i = 2:n-1
    if mod(i,e) == 1 || e == 1
        node_info(i,2:4) = [3,1,1];
    else
        node_info(i,2:4) = [2,0,0];
    end
end
node_info(end,2:4) = [2,1,1];

% 计算总自由度
total_DOF = sum(node_info(:,2));

% 初始化
ub0 = zeros(total_DOF, 1);
vb0 = zeros(total_DOF, 1);



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
n = s0*e+1;
Le = L/e;

node_info = cell(1,e);
node_info{1} = zeros(n,4);
node_info{1}(:,1) = transpose(1:n);
node_info{1}(1,2:4) = [2,1,1];
for i = 2:n-1
    if mod(i,e) == 1 || e == 1
        node_info{1}(i,2:4) = [3,1,1];
    else
        node_info{1}(i,2:4) = [2,0,0];
    end
end
node_info{1}(end,2:4) = [2,1,1];
for j = 2:e
    node_info{j} = node_info{j-1};
    node_info{j}(:,3:4) = [node_info{j-1}(2:end,3:4);0,0];
    for i = 1:n
        if node_info{j}(i,3) == 1
            node_info{j}(i,2) = 3;
        else
            node_info{j}(i,2) = 2;
        end
    end
end

element_info = cell(1,e);
for j = 1:e
    element_info{j} = zeros(s0*e,5);
    element_info{j}(:,1) = transpose(1:s0*e);
    element_info{j}(1,2:5) = [1,2,3,4];
    for i = 2:s0*e
        element_info{j}(i,2) = element_info{j}(i-1,4);
        if node_info{j}(i,4) == 1
            element_info{j}(i,3) = element_info{j}(i,2)+2;
        else
            element_info{j}(i,3) = element_info{j}(i,2)+1;
        end
        element_info{j}(i,4) = element_info{j}(i,3)+1;
        element_info{j}(i,5) = element_info{j}(i,4)+1;
    end
end

Me = mg*Le/420*[156,22*Le,54,-13*Le;22*Le,4*Le^2,13*Le,-3*Le^2;54,13*Le,156,-22*Le;-13*Le,-3*Le^2,-22*Le,4*Le^2];
Ce = rayleighK*EI/Le^3*[12,6*Le,-12,6*Le;6*Le,4*Le^2,-6*Le,2*Le^2;-12,-6*Le,12,-6*Le;6*Le,2*Le^2,-6*Le,4*Le^2]...
    +rayleighM*mg*Le/420*[156,22*Le,54,-13*Le;22*Le,4*Le^2,13*Le,-3*Le^2;54,13*Le,156,-22*Le;-13*Le,-3*Le^2,-22*Le,4*Le^2]...
    -2*mg*V/60*[-30,6*Le,30,-6*Le;-6*Le,0,6*Le,-Le^2;-30,-6*Le,30,6*Le;6*Le,Le^2,-6*Le,0];
Ke = EI/Le^3*[12,6*Le,-12,6*Le;6*Le,4*Le^2,-6*Le,2*Le^2;-12,-6*Le,12,-6*Le;6*Le,2*Le^2,-6*Le,4*Le^2]...
    -rayleighK*EI*V/Le^3*[0,0,0,0;-12,-6*Le,12,-6*Le;0,0,0,0;12,6*Le,-12,6*Le]...
    -rayleighM*mg*V/60*[-30,6*Le,30,-6*Le;-6*Le,0,6*Le,-Le^2;-30,-6*Le,30,6*Le;6*Le,Le^2,-6*Le,0]...
    +mg*V^2/(150*Le)*[-180,-165*Le,180,-15*Le;-15*Le,-20*Le^2,15*Le,5*Le^2;180,15*Le,-180,165*Le;-15*Le,5*Le^2,15*Le,-20*Le^2];

Mb_info = cell(1,e);
Cb_info = cell(1,e);
Kb_info = cell(1,e);
for j = 1:e
    Mb_info{j} = zeros(sum(node_info{j}(:,2)),sum(node_info{j}(:,2)));
    Cb_info{j} = zeros(sum(node_info{j}(:,2)),sum(node_info{j}(:,2)));
    Kb_info{j} = zeros(sum(node_info{j}(:,2)),sum(node_info{j}(:,2)));
    for i = 1:s0*e
        Mb_info{j}(element_info{j}(i,2:5),element_info{j}(i,2:5)) = Mb_info{j}(element_info{j}(i,2:5),element_info{j}(i,2:5))+Me;
        Cb_info{j}(element_info{j}(i,2:5),element_info{j}(i,2:5)) = Cb_info{j}(element_info{j}(i,2:5),element_info{j}(i,2:5))+Ce;
        Kb_info{j}(element_info{j}(i,2:5),element_info{j}(i,2:5)) = Kb_info{j}(element_info{j}(i,2:5),element_info{j}(i,2:5))+Ke;
    end
    for i = 1:n
        dof = sum(node_info{j}(1:i-1,2))+1;
        if node_info{j}(i,3) == 1 && i ~= 1 && i ~= n
            Kb_info{j}(dof,dof) = Kb_info{j}(dof,dof)+2*kb;
        end
        if node_info{j}(i,4) == 1 && i ~= 1 && i ~= n
            Kb_info{j}([dof+1,dof+2],[dof+1,dof+2]) = Kb_info{j}([dof+1,dof+2],[dof+1,dof+2])+[kc,-kc;-kc,kc];
        end
    end
    Kb_info{j}(1,1) = Kb_info{j}(1,1)+10^20;
    Kb_info{j}(2,2) = Kb_info{j}(2,2)+10^20;
    Kb_info{j}(end-1,end-1) = Kb_info{j}(end-1,end-1)+10^20;
    Kb_info{j}(end,end) = Kb_info{j}(end,end)+10^20;
end

% Numerical Integration
dt = Le/V;
t = 0:dt:floor(s*L/(V*dt))*dt;
yv = zeros(10,length(t));
vv = zeros(10,length(t));
av = zeros(10,length(t));
ub = cell(1,length(t));
vb = cell(1,length(t));
ab = cell(1,length(t));

yv(:,1) = yv0;
vv(:,1) = vv0;
ub{1} = ub0;
vb{1} = vb0;

Mb = Mb_info{1};
Cb = Cb_info{1};
Kb = Kb_info{1};

train_pos = l*[7,5,3,1,-1,-3,-5,-7];
Pv0 = zeros(10,1);
Pv0(1) = mc*9.81;
for j = 1:8
    element_cur = ceil(train_pos(j)/Le)+s0*e/2;
    x = mod(train_pos(j),Le);
    N1 = 1-3*(x/Le)^2+2*(x/Le)^3;
    N2 = ((x/Le)-2*(x/Le)^2+(x/Le)^3)*Le;
    N3 = 3*(x/Le)^2-2*(x/Le)^3;
    N4 = (-(x/Le)^2+(x/Le)^3)*Le;
    dof1 = element_info{1}(element_cur,2);
    dof2 = element_info{1}(element_cur,3);
    dof3 = element_info{1}(element_cur,4);
    dof4 = element_info{1}(element_cur,5);
    u1 = ub{1}(dof1);
    theta1 = ub{1}(dof2);
    u2 = ub{1}(dof3);
    theta2 = ub{1}(dof4);
    u = [N1,N2,N3,N4]*[u1;theta1;u2;theta2];
    Pv0(j+2) = mm*9.81-F0+(Ci*kh-Ch)*(u+h0);
end
av(:,1) = Mv\(Pv0-Cv*vv(:,1)-Kv*yv(:,1));
Pb0 = zeros(length(ub{1}),1);
for j = 1:8
    element_cur = ceil(train_pos(j)/Le)+s0*e/2;
    x = mod(train_pos(j),Le);
    N1 = 1-3*(x/Le)^2+2*(x/Le)^3;
    N2 = ((x/Le)-2*(x/Le)^2+(x/Le)^3)*Le;
    N3 = 3*(x/Le)^2-2*(x/Le)^3;
    N4 = (-(x/Le)^2+(x/Le)^3)*Le;
    dof1 = element_info{1}(element_cur,2);
    dof2 = element_info{1}(element_cur,3);
    dof3 = element_info{1}(element_cur,4);
    dof4 = element_info{1}(element_cur,5);
    F = mm*9.81-mm*av(j+2,1)-c*(vv(j+2,1)-(vv(1,1)+Lc(j)*vv(2,1)))-k*(yv(j+2,1)-(yv(1,1)+Lc(j)*yv(2,1)));
    Pb0([dof1,dof2,dof3,dof4]) = Pb0([dof1,dof2,dof3,dof4])+F*[N1;N2;N3;N4];
end
ab{1} = Mb\(Pb0-Cb*vb{1}-Kb*ub{1});

beta = 0.25;
gamma = 0.5;
a1 = 1/(beta*dt^2);
a2 = 1/(beta*dt);
a3 = 1/(2*beta)-1;
b1 = gamma/(beta*dt);
b2 = gamma/beta-1;
b3 = dt*(gamma/(2*beta)-1);

Kv_eff = Kv+a1*Mv+b1*Cv;

for i = 2:length(t)
    Mb = Mb_info{mod(i-1,e)+1};
    Cb = Cb_info{mod(i-1,e)+1};
    Kb = Kb_info{mod(i-1,e)+1};
    Kb_eff = Kb+a1*Mb+b1*Cb;

    if mod(i-1,e)+1 == 1
        ub_init = [ub{i-1}([3,5:end]);0;0];
        vb_init = [vb{i-1}([3,5:end]);0;0];
        ab_init = [ab{i-1}([3,5:end]);0;0];
    elseif mod(i-1,e)+1 == 2
        ub_init = [ub{i-1}(3:end);ub{i-1}(end);0;0];
        vb_init = [vb{i-1}(3:end);vb{i-1}(end);0;0];
        ab_init = [ab{i-1}(3:end);ab{i-1}(end);0;0];
    else
        ub_init = [ub{i-1}(3:end);0;0];
        vb_init = [vb{i-1}(3:end);0;0];
        ab_init = [ab{i-1}(3:end);0;0];
    end

    ub_prev = ub_init;

    err = Inf;
    while err > 1e-8
        ub_cur = ub_prev;

        vb_cur = b1*(ub_cur-ub_init)-b2*vb_init-b3*ab_init;
        ab_cur = a1*(ub_cur-ub_init)-a2*vb_init-a3*ab_init;

        Pv = zeros(10,1);
        Pv(1) = mc*9.81;
        for j = 1:8
            element_cur = ceil(train_pos(j)/Le)+s0*e/2;
            x = mod(train_pos(j),Le);
            N1 = 1-3*(x/Le)^2+2*(x/Le)^3;
            N2 = ((x/Le)-2*(x/Le)^2+(x/Le)^3)*Le;
            N3 = 3*(x/Le)^2-2*(x/Le)^3;
            N4 = (-(x/Le)^2+(x/Le)^3)*Le;
            dof1 = element_info{mod(i-1,e)+1}(element_cur,2);
            dof2 = element_info{mod(i-1,e)+1}(element_cur,3);
            dof3 = element_info{mod(i-1,e)+1}(element_cur,4);
            dof4 = element_info{mod(i-1,e)+1}(element_cur,5);
            u1 = ub_cur(dof1);
            theta1 = ub_cur(dof2);
            u2 = ub_cur(dof3);
            theta2 = ub_cur(dof4);
            u = [N1,N2,N3,N4]*[u1;theta1;u2;theta2];
            Pv(j+2) = mm*9.81-F0+(Ci*kh-Ch)*(u+h0);
        end

        Pv_eff = Pv+(a1*Mv+b1*Cv)*yv(:,i-1)+(a2*Mv+b2*Cv)*vv(:,i-1)+(a3*Mv+b3*Cv)*av(:,i-1);
        yv_cur = Kv_eff\Pv_eff;

        vv_cur = b1*(yv_cur-yv(:,i-1))-b2*vv(:,i-1)-b3*av(:,i-1);
        av_cur = a1*(yv_cur-yv(:,i-1))-a2*vv(:,i-1)-a3*av(:,i-1);

        Pb = zeros(length(ub_cur),1);
        for j = 1:8
            element_cur = ceil(train_pos(j)/Le)+s0*e/2;
            x = mod(train_pos(j),Le);
            N1 = 1-3*(x/Le)^2+2*(x/Le)^3;
            N2 = ((x/Le)-2*(x/Le)^2+(x/Le)^3)*Le;
            N3 = 3*(x/Le)^2-2*(x/Le)^3;
            N4 = (-(x/Le)^2+(x/Le)^3)*Le;
            dof1 = element_info{mod(i-1,e)+1}(element_cur,2);
            dof2 = element_info{mod(i-1,e)+1}(element_cur,3);
            dof3 = element_info{mod(i-1,e)+1}(element_cur,4);
            dof4 = element_info{mod(i-1,e)+1}(element_cur,5);
            F = mm*9.81-mm*av_cur(j+2)-c*(vv_cur(j+2)-(vv_cur(1)+Lc(j)*vv_cur(2)))-k*(yv_cur(j+2)-(yv_cur(1)+Lc(j)*yv_cur(2)));
            Pb([dof1,dof2,dof3,dof4]) = Pb([dof1,dof2,dof3,dof4])+F*[N1;N2;N3;N4];
        end

        Pb_eff = Pb+(a1*Mb+b1*Cb)*ub_init+(a2*Mb+b2*Cb)*vb_init+(a3*Mb+b3*Cb)*ab_init;
        ub_cur = Kb_eff\Pb_eff;

        err = norm(ub_cur-ub_prev)/norm(ub_cur-ub_init);

        ub_prev = ub_cur;
    end

    yv(:,i) = yv_cur;
    vv(:,i) = vv_cur;
    av(:,i) = av_cur;
    ub{i} = ub_cur;
    vb{i} = vb_cur;
    ab{i} = ab_cur;
end
end