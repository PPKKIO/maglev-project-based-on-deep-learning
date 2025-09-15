% Maglev Model 1
% 1-DOF Train on Rigid Guideway
% Analytical Solution

% Train Parameters
% mv = mass of train
% i0 = nominal current
% h0 = nominal air gap
% kp = feedback current gain (air gap)
% kh = feedback current gain (velocity)
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

function [t,yv,vv,av] = solve_model1_anal(mv,i0,h0,kh,kv,ka,yv0,vv0,dt,t_end)
K0 = mv*9.81/(i0/h0)^2;
Ci = 2*K0*i0/h0^2;
Ch = 2*K0*i0^2/h0^3;

Mv = mv+Ci*ka;
Cv = Ci*kv;
Kv = Ci*kh-Ch;

t = 0:dt:t_end;

zeta = Cv/(2*sqrt(Mv*Kv));

if zeta < 1 % Underdamping
    omega0 = sqrt(Kv/Mv);
    omega = omega0*sqrt(1-zeta^2);
    A = yv0-h0;
    B = (vv0+zeta*omega0*(yv0-h0))/omega;
    yv = sqrt(A^2+B^2)*exp(-zeta*omega0*t).*cos(omega*t-atan2(B,A))+h0;
    vv = -sqrt(A^2+B^2)*sqrt((zeta*omega0)^2+omega^2)*exp(-zeta*omega0*t).*cos(omega*t-atan2(B,A)-atan2(omega,zeta*omega0));
    av = sqrt(A^2+B^2)*((zeta*omega0)^2+omega^2)*exp(-zeta*omega0*t).*cos(omega*t-atan2(B,A)-2*atan2(omega,zeta*omega0));
elseif zeta > 1 % Overdamping
    lambda1 = (-Cv-sqrt(Cv^2-4*Mv*Kv))/(2*Mv);
    lambda2 = (-Cv+sqrt(Cv^2-4*Mv*Kv))/(2*Mv);
    A = (lambda2*(yv0-h0)-vv0)/(lambda2-lambda1);
    B = (-lambda1*(yv0-h0)+vv0)/(lambda2-lambda1);
    yv = A*exp(lambda1*t)+B*exp(lambda2*t)+h0;
    vv = A*lambda1*exp(lambda1*t)+B*lambda2*exp(lambda2*t);
    av = A*lambda1^2*exp(lambda1*t)+B*lambda2^2*exp(lambda2*t);
elseif zeta == 1 % Critical Damping
    lambda = -Cv/(2*Mv);
    A = yv0-h0;
    B = vv0-lambda*(yv0-h0);
    yv = A*exp(lambda*t)+B*t.*exp(lambda*t);
    vv = (A*lambda+B)*exp(lambda*t)+B*lambda*t.*exp(lambda*t);
    av = (A*lambda^2+2*B*lambda)*exp(lambda*t)+B*lambda^2*t.*exp(lambda*t);
end
end