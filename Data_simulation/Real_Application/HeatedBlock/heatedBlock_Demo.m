% 


% we can find the maximum or minimum of function value xtime = heatedBlock2(lx,ly,theta, Hmax) 
% range
% 0.1<lx<0.4
% 0.1<ly<0.4
% 0<theta<2*pi
% the optimal shoud be lx = 0.1, ly = 0.1. and theta = any value, but this
% has not been confirmed.
% we can also set 0.1<lx<0.4 and 0.2<ly<0.4.


% for low-fidelity 
% Hmax = 0.8;

X = [0.1,0.1,pi/2];
% xtime = heatedBlock2(0.1,0.1,pi/2, Hmax)
y_lf = HeatedBlockQuerySilent(X,0)


% for high-fidelity 
Hmax = 0.2;

X = [0.1,0.1,0];
y_hf = HeatedBlockQuerySilent(X, 1)