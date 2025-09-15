% Maglev Model 2
% 1-DOF Train on Flexible Guideway
% Moving Element Method

% Train Parameters
% mv = mass of train
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

function [t,yv,vv,av] = solve_model2_mem(mv,i0,h0,kh,kv,ka,V, ...
    L,mg,EI,kb,kc,rayleighM,rayleighK,s,e, ...
    yv0,vv0,ub0,vb0,s0)

% Train
K0 = mv*9.81/(i0/h0)^2;
Ci = 2*K0*i0/h0^2;
Ch = 2*K0*i0^2/h0^3;

Mv = mv+Ci*ka;
Cv = Ci*kv;
Kv = Ci*kh-Ch;

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
yv = zeros(1,length(t));
vv = zeros(1,length(t));
av = zeros(1,length(t));
ub = cell(1,length(t));
vb = cell(1,length(t));
ab = cell(1,length(t));

yv(1) = yv0;
vv(1) = vv0;
ub{1} = ub0;
vb{1} = vb0;

Mb = Mb_info{1};
Cb = Cb_info{1};
Kb = Kb_info{1};

Pv0 = (Ci*kh-Ch)*(ub{1}(floor(end/2))+h0);
av(1) = Mv\(Pv0-Cv*vv(1)-Kv*yv(1));
F = mv*9.81-mv*av(1);
Pb0 = zeros(length(ub{1}),1);
Pb0(floor(end/2)) = F;
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
        
        u = ub_cur(floor(end/2));
        Pv = (Ci*kh-Ch)*(u+h0);

        Pv_eff = Pv+(a1*Mv+b1*Cv)*yv(i-1)+(a2*Mv+b2*Cv)*vv(i-1)+(a3*Mv+b3*Cv)*av(i-1);
        yv_cur = Kv_eff\Pv_eff;

        vv_cur = b1*(yv_cur-yv(i-1))-b2*vv(i-1)-b3*av(i-1);
        av_cur = a1*(yv_cur-yv(i-1))-a2*vv(i-1)-a3*av(i-1);

        F = mv*9.81-mv*av_cur;
        Pb = zeros(length(ub_cur),1);
        Pb(floor(end/2)) = F;

        Pb_eff = Pb+(a1*Mb+b1*Cb)*ub_init+(a2*Mb+b2*Cb)*vb_init+(a3*Mb+b3*Cb)*ab_init;
        ub_cur = Kb_eff\Pb_eff;

        err = norm(ub_cur-ub_prev)/norm(ub_cur-ub_init);

        ub_prev = ub_cur;
    end

    yv(i) = yv_cur;
    vv(i) = vv_cur;
    av(i) = av_cur;
    ub{i} = ub_cur;
    vb{i} = vb_cur;
    ab{i} = ab_cur;
end
end