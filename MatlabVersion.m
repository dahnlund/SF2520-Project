%Project Catalytic combustion GROUP 2
clear all, clc

%GIVEN DATA:
eta=0.2;
gamma=100;
alpha=0.2;
w=0.3;

%DISCRETIZATION:
M=1000;
dz=1/M;
N=round(w/dz); %Make sure not to choose M so that N isnt an integer

zv=dz:dz:1-dz; %JUST for Velocity in gas-region
v = @(z) 1-4*(z-(1/2)).^2;

%Creates A1 matrix
e = ones(M-1,1).*(eta./((dz^2)*v(zv)'));
A1 = spdiags([[e(2:end);e(1)] -2*e [e(1);e(1:end-1);]], -1:1, M-1, M-1);
A1(1,1)=A1(1,1)/3; A1(1,2)=A1(1,2)*2/3; %Change boundary

%Matrices/vectors inbetween
e1=[zeros(M-2,1);eta/(v(zv(end))*dz^2)];
b1=[zeros(1,M-2) -1/dz];
a=(1+alpha)/dz;
b2=[-alpha/dz zeros(1,N-2)];
e2=[1/dz^2;zeros(N-2,1)];

%Creates A2 matrix
e = ones(N-1,1)/(dz^2);
A2 = spdiags([e -2*e-gamma e], -1:1, N-1, N-1);
A2(end,end)=((-2/(3*dz^2))-gamma); A2(end,end-1)=(2/(3*dz^2));

Atot=[A1 e1 zeros(M-1,N-1);b1 a b2; zeros(N-1,M-1) e2 A2];

%We choose to Ñ=N-1 and then add the last boundary value

%Implicit Euler Method

dt=0.001;
t=dt:dt:1;
uVec=[];uVec(:,1)=[ones(M-1,1);0;zeros(N-1,1)];
B=Atot;
for i=t
    u_new=B\(uVec(:,end).*(1/dt));
    uVec=[uVec u_new];
end
t=0:dt:1;
z=dz:dz:1+w;

size(t)
size(z)
size(uVec)
mesh(z(1:end-1),t,uVec')
