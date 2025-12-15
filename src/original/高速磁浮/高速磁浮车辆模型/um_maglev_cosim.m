function [sys,x0,str,ts] =高速磁浮车辆模型(t,x,u,flag)
switch flag,
case 0
[sys,x0,str,ts] = mdlInitializeSizes();
case 3
sys = mdlOutputs(t,x,u);
case {1, 2, 4}
sys = [];
case 9;
sys = mdlTerminate(t,x,u);
otherwise
error(['Unhandled flag = ',num2str(flag)]);
end;

function [sys,x0,str,ts] = mdlInitializeSizes()
sizes = simsizes;
sizes.NumContStates= 0;
sizes.NumDiscStates= 0;
sizes.NumOutputs=1;
sizes.NumInputs=5;
sizes.DirFeedthrough=1;
sizes.NumSampleTimes=1;
sys = simsizes(sizes);
x0 = []; % No continuous states
str = []; % No state ordering
ts = [0]; % Inherited sample time
global h0
global errcode
global erroutput
global errinit
global errloadcode
global errload
erroutput=false;
errinit=false;
errcode=0;
errloadcode=0;
errload=false;
if (iscom(h0))==0
  h0=actxserver('UMCoSimulation.UMMatlabCoSimul');
  errloadcode=h0.LoadObjectFromFile('c:\users\10301\desktop\磁悬浮列车项目\人工智能项目\src\original\高速磁浮\高速磁浮车辆模型\input.dat');
  if ~(errloadcode==0)
    errload=true;
    errordlg('UM COM Server returns the following error message: "Loading model fails". Simulation is interrupted.','UM COM Server error');
  end
  if ~errload
    h0.LoadMatlabSettings('c:\users\10301\desktop\磁悬浮列车项目\人工智能项目\src\original\高速磁浮\高速磁浮车辆模型\um_maglev_cosim.cosim');
    h0.ReadTotalConfiguration('c:\users\10301\desktop\磁悬浮列车项目\人工智能项目\src\original\高速磁浮\高速磁浮车辆模型\um_maglev_cosim');
    errcode=h0.PrepareIntegration();
  end
end
if errcode==0
elseif (errcode==1)
  errinit=true;
  errordlg('UM COM Server returns the following error message: "Loading model fails". Simulation is interrupted.','UM COM Server error');
else
  errinit=true;
  errordlg(strcat('UM COM Server returns the following error message: "', h0.GetLastError(), '". Simulation is interrupted.'), 'UM COM Server error');
end
% End of mdlInitializeSizes.

function sys = mdlOutputs(t,x,u)
global h0
global errcode
global erroutput
global errinit
global errload
if errinit || errload set_param(gcs, 'SimulationCommand', 'stop'); sys=[];
elseif erroutput==false
  value=[u(1) u(2) u(3) u(4) u(5)];
  h0.SetValues(value);
  errcode=h0.DoIntegrationInterval(t);
  if errcode==0
  elseif (errcode==1)
    erroutput=true;
    set_param(gcs, 'SimulationCommand', 'stop');
    errordlg('UM COM Server returns the following error message: "Server is busy". Simulation is interrupted.','UM COM Server error');
  elseif (errcode==2)
    erroutput=true;
    set_param(gcs, 'SimulationCommand', 'stop');
    errordlg('UM COM Server returns the following error message: "Preparing integration fails". Simulation is interrupted.','UM COM Server error');
  else
    erroutput=true;
    set_param(gcs, 'SimulationCommand', 'stop');
    errordlg(strcat('UM COM Server returns the following error message: "', h0.GetLastError(), '". Simulation is interrupted.'), 'UM COM Server error');
  end
  sys=h0.GetValues();
else sys=[];
end
% End of mdlOutputs.

function sys = mdlTerminate(t,x,u)
global h0
global errload
global errinit
if (~errload) && (~errinit)
  h0.FinishIntegration();
end
h0.delete;
sys=[];
% End of mdlTerminate.
