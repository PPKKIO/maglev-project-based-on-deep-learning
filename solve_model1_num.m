% Maglev Model 1
% 1-DOF Train on Rigid Guideway
% Numerical Integration Using Newmark's Method

% Train Parameters
% mv = mass of train
% i0 = nominal current
% h0 = nominal air gap
% kh = feedback current gain (air gap)
% kv = feedback current gain (velocity)
% ka = feedback current gain (acceleration)

% yv0 = initial train displacement
% vv0 = initial train velocity

% Other Inputs
% dt = time step
% t_end = final time value

% Outputs
% t = time vector (for plotting of graphs)
% yv = train displacement
% vv = train velocity
% av = train acceleration

function [t,yv,vv,av] = solve_model1_num(mv,i0,h0,kh,kv,ka,yv0,vv0,dt,t_end)
K0 = mv*9.81/(i0/h0)^2;
Ci = 2*K0*i0/h0^2;
Ch = 2*K0*i0^2/h0^3;

Mv = mv+Ci*ka;
Cv = Ci*kv;
Kv = Ci*kh-Ch;

t = 0:dt:t_end;
yv = zeros(1,length(t));
vv = zeros(1,length(t));
av = zeros(1,length(t));

yv(1) = yv0;
vv(1) = vv0;

Pv0 = (Ci*kh-Ch)*h0;
av0 = Mv^-1*(Pv0-Cv*vv0-Kv*yv0);
av(1) = av0;

beta = 0.25;
gamma = 0.5;
a1 = 1/(beta*dt^2);
a2 = 1/(beta*dt);
a3 = 1/(2*beta)-1;
b1 = gamma/(beta*dt);
b2 = gamma/beta-1;
b3 = dt*(gamma/(2*beta)-1);

for i = 2:length(t)
    Pv = (Ci*kh-Ch)*h0;

    Kv_eff = Kv+a1*Mv+b1*Cv;
    Pv_eff = Pv+(a1*Mv+b1*Cv)*yv(i-1)+(a2*Mv+b2*Cv)*vv(i-1)+(a3*Mv+b3*Cv)*av(i-1);
    yv(i) = Kv_eff^-1*Pv_eff;

    vv(i) = b1*(yv(i)-yv(i-1))-b2*vv(i-1)-b3*av(i-1);
    av(i) = a1*(yv(i)-yv(i-1))-a2*vv(i-1)-a3*av(i-1);
end
end