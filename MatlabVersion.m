%Project Catalytic combustion GROUP 2
clear all, clc

%GIVEN DATA:
eta=0.2;
gamma=100;
alpha=0.2;
w=0.3;

%DISCRETIZATION:
M=10;
dz=1/M;
N=round(w/dz); %Make sure not to choose M so that N isnt an integer


zv=dz:dz:1-dz; %JUST for Velocity in gas-region
v = @(z) 1-4*(z-(1/2)).^2;

%Creates A1 matrix
e = ones(M-1,1).*(eta./((dz^2)*v(zv)'));
A1 = spdiags([[e(2:end);e(1)] -2*e [e(1);e(1:end-1);]], -1:1, M-1, M-1);
A1(1,1)=A1(1,1)/3;
A1(1,2)=A1(1,2)*2/3;

%Matrices/vectors inbetween
e1=[zeros(M-2,1);eta/(v(zv(end))*dz^2)];
b1=[zeros(1,M-2) -1/dz];
a=(1+alpha)/dz;
b2=[-alpha/dz zeros(1,N-1)];
e2=[1/dz^2;zeros(N-1,1)];

%Creates A2 matrix
e = ones(N,1)/(dz^2);
A2 = spdiags([e -(2+gamma*dz^2)*e e], -1:1, N, N);
A2(end,end-1)=4/3;
A2(end,end-2)=-1/3;
A2(end,end)=0;

Atot=[A1 e1 zeros(M-1,N);b1 a b2; zeros(N,M-1) e2 A2];



