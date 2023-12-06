%Project Catalytic combustion GROUP 2
clear all, clc

%GIVEN DATA:
eta=0.2;
gamma=100;
alpha=0.2;
w=0.3;

%% DISCRETIZATION:
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
b2=[-alpha/dz zeros(1,N-1)];
e2=[1/dz^2;zeros(N-1,1)];

%Creates A2 matrix
e = ones(N,1)/(dz^2);
A2 = spdiags([e -2*e-gamma e], -1:1, N, N);
A2(end,end)=((-2/(3*dz^2))-gamma); A2(end,end-1)=(2/(3*dz^2)); A2(end-1,end)=0;

Atot=[A1 e1 zeros(M-1,N);b1 a b2; zeros(N,M-1) e2 A2];

%% Implicit Euler Method
dt=0.001;
uVec=[]; uVec(:,1)=[ones(M-1,1);0;zeros(N,1)]; %Starting values=1 for u_g

eyeUg=[eye(M-1) zeros(M-1,N+1);zeros(N+1,M+N)];
B=[eyeUg-dt*Atot];

t=dt:dt:1;
for i=t
    u_new=B\(eyeUg*uVec(:,end));
    uVec=[uVec u_new];
end
uVec(1,end)

%Plots the total uVec
t=0:dt:1;
z=dz:dz:1+w;
figure(1)
mesh(z,t,uVec')
xlabel("Z")
ylabel("T")
zlabel("U")
title("How u changes over tau (Implicit Euler)")

%% Regularization and Implicit Euler
epsilon=0.01;
epsiEye=spdiags([ones(M-1,1);(1/epsilon);ones(N,1)*(1/epsilon)],0,M+N,M+N);
AtotReg=epsiEye*Atot;

uVec=[]; uVec(:,1)=ones(M+N,1); %Change this for different starting values
B=[eye(M+N)-dt*AtotReg];
t=dt:dt:1;
for i=t
    u_new=B\uVec(:,end);
    uVec=[uVec u_new];
end
uVec(1,end)

%Plots the total uVec
t=0:dt:1;
z=dz:dz:1+w;
figure(2)
mesh(z,t,uVec')
xlabel("Z")
ylabel("T")
zlabel("U")
title("How u changes over tau (Regularization+Implicit Euler)")

%% Analytic reduction and Implicit Euler

%beta=alpha*sqrt(gamma)*((1-(1/exp(2*w*sqrt(gamma))))/((1/exp(2*w*sqrt(gamma)))+1))
%Beta above ^ is the same as:
beta=alpha*sqrt(gamma)*tanh(w*sqrt(gamma));

%Since we are just solving the PDE(which has a new boundary condition at Z=1)
e = ones(M-1,1).*(eta./((dz^2)*v(zv)'));
A1 = spdiags([[e(2:end);e(1)] -2*e [e(1);e(1:end-1);]], -1:1, M-1, M-1);
A1(1,1)=A1(1,1)/3; A1(1,2)=A1(1,2)*2/3; %Change boundary
%A1(end,end)=(eta*((1/(1+dz*beta))-2))/(v(zv(end)*dz^2)); %The new boundary condition
A1(end,end)=A1(end,end)*(1-1/(2*(1+dz*beta)));

uVec=[]; uVec(:,1)=ones(M-1,1); %Change this for different starting values

B=[eye(M-1)-dt*A1];
t=dt:dt:1;
for i=t
    u_new=B\uVec(:,end);
    uVec=[uVec u_new];
end
uVec(1,end)
%Plots the total uVec
t=0:dt:1;
z=dz:dz:1-dz;

figure(3)
mesh(z,t,uVec')
xlabel("Z")
ylabel("T")
zlabel("U")
title("How u changes over tau (Analytic reduction+Implicit Euler)")