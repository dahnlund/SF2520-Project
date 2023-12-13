% Matlab Version:
% Project Catalytic combustion GROUP 2 SF2520
% Max Hollingworth, Rikard Landfeldt, David Ahnlund, Albert Sund Aillet

clear all, clc, close all

%GIVEN DATA:
eta=0.2;
gamma=100;
alpha=0.2;
w=0.3;

%% DISCRETIZATION:
M=100; %Discretization of z-axis
dz=1/M;
N=round(w/dz);

zv=dz:dz:1-dz; %JUST for Velocity in gas-region
v = @(z) 1-4*(z-(1/2)).^2;

%Creates A1 matrix
e = ones(M-1,1).*(eta./((dz^2)*v(zv)'));
A1 = spdiags([[e(2:end);e(1)] -2*e [e(1);e(1:end-1);]], -1:1, M-1, M-1);
A1(1,1)=A1(1,1)/3; A1(1,2)=A1(1,2)*2/3; %Change according to left boundary

%Matrices/vectors "inbetween" A1 and A2
e1=[zeros(M-2,1);eta/(v(zv(end))*dz^2)];
b1=[zeros(1,M-2) -1/dz];
a=(1+alpha)/dz;
b2=[-alpha/dz zeros(1,N-1)];
e2=[1/dz^2;zeros(N-1,1)];

%Creates A2 matrix
e = ones(N,1)/(dz^2);
A2 = spdiags([e -2*e-gamma e], -1:1, N, N);
A2(end,end)=1/(dz^2)*(-2/3)-gamma;
A2(end,end-1)=1/(dz^2)*(2/3);

%The entire matrix put together
Atot=[A1 e1 zeros(M-1,N);b1 a b2; zeros(N,M-1) e2 A2];



dt=0.01; %Discretization of tau-axis

%% Implicit Euler Method
uVec=[]; uVec(:,1)=[ones(M-1,1);0;zeros(N,1)]; %Starting values=1 for u_g

eyeUg=sparse([eye(M-1) zeros(M-1,N+1);zeros(N+1,M+N)]); %Creates "identity matrix" for ug values only
%B=sparse([sparse(eye(M+N))-dt*Atot]); %Creates LHS of implicit euler equation
B=sparse([eyeUg-dt*Atot]); %Creates LHS of implicit euler equation
B=decomposition(B);

t=dt:dt:1;
tic
for i=t
    u_new=B\[uVec(1:M-1,end);zeros(N+1,1)];
    uVec=[uVec u_new];
end
time=toc;
disp("Time Implicit Euler method: " + time + " s")
impUVec=uVec;
%Plots the total uVec
t=0:dt:1;
z=dz:dz:1+w;
figure(1)
% plot(z,uVec(:,find(t==0)),z,uVec(:,find(t==0.1)),z,uVec(:,find(t==0.2)),z,uVec(:,find(t==0.5)),z,uVec(:,find(t==0.7)),z,uVec(:,find(t==0.9)),z,uVec(:,find(t==1)))
% xlabel("z")
% ylabel("u")
% legend("τ=0","τ=0.1","τ=0.2","τ=0.5","τ=0.7","τ=0.9","τ=1")
% title("u over z for different τ (Implicit Euler Method)")
mesh(z,t,uVec')
xlim([0 1.3])
ylim([0 1])
zlim([0 1.2])
xlabel("Z")
ylabel("τ")
zlabel("U")
title("How u changes over tau (Implicit Euler)")

%% Regularization and Implicit Euler
epsilon=1e-3; %Choose a smaller value for more accuracy
epsiEye=spdiags([ones(M-1,1);(1/epsilon);ones(N,1)*(1/epsilon)],0,M+N,M+N);
AtotReg=epsiEye*Atot;

uVec=[]; uVec(:,1)=ones(M+N,1); %Change this for different starting values
%B=sparse([sparse(eye(M+N))-dt*AtotReg]);
B=sparse([eyeUg-dt*AtotReg]); %Creates LHS of implicit euler equation

B=decomposition(B);

t=dt:dt:1;
tic
for i=t
    u_new=B\uVec(:,end);
    uVec=[uVec u_new];
end
time=toc;
disp("Time Regularization + Implicit Euler method: " + time + " s")

regUVec=uVec;

%Plots the total uVec
t=0:dt:1;
z=dz:dz:1+w;
figure(2)
%plot(z,uVec(:,1),z,uVec(:,find(t==0.001)),z,uVec(:,find(t==0.2)),z,uVec(:,find(t==0.5)),z,uVec(:,find(t==0.8)),z,uVec(:,find(t==1)))
mesh(z,t,uVec')
xlim([0 1.3])
ylim([0 1])
zlim([0 1.2])
xlabel("Z")
ylabel("τ")
zlabel("U")
title("How u changes over tau (Regularization+Implicit Euler)")

%% Analytic reduction and Implicit Euler

%beta=alpha*sqrt(gamma)*((1-(1/exp(2*w*sqrt(gamma))))/((1/exp(2*w*sqrt(gamma)))+1))
%Beta above ^ is the same as:
beta=alpha*sqrt(gamma)*tanh(w*sqrt(gamma));

%A1(end,end)=A1(end,end)*(1-1/(2*(1+dz*beta)));
A1(end,end)=A1(end,end)*(1-(2/(3+beta*dz*2)));
A1(end,end-1)=A1(end,end-1)*(1-(1/(3+beta*2*dz)));

uVec=[]; uVec(:,1)=ones(M-1,1); %Change this for different starting values

B=sparse([sparse(eye(M-1))-dt*A1]);
B=decomposition(B);
t=dt:dt:1;
tic
for i=t
    u_new=B\uVec(:,end);
    uVec=[uVec u_new];
end
time=toc;
disp("Time Analyctic-red + Implicit Euler method: " + time + " s")

anaUVec=uVec;

%Plots the total uVec
t=0:dt:1;
z=dz:dz:1-dz;

figure(3)
mesh(z,t,uVec')
xlim([0 1.3])
ylim([0 1])
zlim([0 1.2])
xlabel("Z")
ylabel("τ")
zlabel("U")
title("How u changes over tau (Analytic reduction+Implicit Euler)")


%% Plots all meshes togheter
figure(4)

t=0:dt:1;
z=dz:dz:1+w;
mesh(z,t,impUVec','EdgeColor','g')
hold on
mesh(z,t,regUVec','EdgeColor','r')
mesh(z(1:M-1),t,anaUVec','EdgeColor','b')
hold off

%% Investigation of Solution
%---------------------------------------------------------------
%DATA TO BE CHANGED:
eta=0.2;
gamma=100;
alpha=0.2;
w=0.3;
%---------------------------------------------------------------

%Since we just want to calculate the integral of ug not us we can use the
%analytical reduction solutuion which only has ug
%Using trapz to calculate the integral from z in (0:1) at different lengths tau
%Since it will depend on the size of M, we can just take the percentage (aka divide by M)
[gasPerVec,z]=investigation(eta,gamma,alpha,w);
figure(6)
plot(z,gasPerVec)

%We can change the parameters:
etaVec=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]';
gammaVec=[25 50 100 250 500 750 1000]';
alphaVec=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]';
wVec=[0.10 0.15 0.20 0.25 0.30 0.35 0.40]';

figure(7)
hold on
for i=1:length(etaVec)
    [gasPerVec,z]=investigation(etaVec(i),gamma,alpha,w);
    plot(z,gasPerVec)
end
hold off
legend("\eta =" + etaVec,'Location','southwest')
title("Concentration of fuel when changing \eta")
xlabel("Z")
ylabel("T(\tau)")

figure(8)
hold on
for i=1:length(gammaVec)
    [gasPerVec,z]=investigation(eta,gammaVec(i),alpha,w);
    plot(z,gasPerVec)
end
hold off
legend("\gamma =" + gammaVec,'Location','southwest')
title("Concentration of fuel when changing \gamma")
xlabel("Z")
ylabel("T(\tau)")

figure(9)
hold on
for i=1:length(alphaVec)
    [gasPerVec,z]=investigation(eta,gamma,alphaVec(i),w);
    plot(z,gasPerVec)
end
hold off
legend("\alpha =" + alphaVec,'Location','southwest')
title("Concentration of fuel when changing \alpha")
xlabel("Z")
ylabel("T(\tau)")

figure(10)
hold on
for i=1:length(wVec)
    [gasPerVec,z]=investigation(eta,gamma,alpha,wVec(i));
    plot(z,gasPerVec)
end
hold off
legend("w =" + wVec,'Location','southwest')
title("Concentration of fuel when changing w")
xlabel("Z")
ylabel("T(\tau)")

%%
function [gasPerVec,z]=investigation(eta,gamma,alpha,w) %Investigate with analytical solution
    M=1000; %Discretization of z-axis
    dz=1/M;
    N=round(w/dz);
    zv=dz:dz:1-dz; %JUST for Velocity in gas-region
    v = @(z) 1-4*(z-(1/2)).^2;
    %Creates A1 matrix
    e = ones(M-1,1).*(eta./((dz^2)*v(zv)'));
    A1 = spdiags([[e(2:end);e(1)] -2*e [e(1);e(1:end-1);]], -1:1, M-1, M-1);
    A1(1,1)=A1(1,1)/3; A1(1,2)=A1(1,2)*2/3; %Change according to left boundary
    beta=alpha*sqrt(gamma)*tanh(w*sqrt(gamma));
    A1(end,end)=A1(end,end)*(1-(2/(3+beta*dz*2)));
    A1(end,end-1)=A1(end,end-1)*(1-(1/(3+beta*2*dz)));

    dt=0.001; %Discretization of tau-axis

    uVec=[]; uVec(:,1)=ones(M-1,1); %Change this for different starting values
    B=sparse([sparse(eye(M-1))-dt*A1]);
    B=decomposition(B);
    t=dt:dt:1;
    for i=t
        u_new=B\uVec(:,end);
        uVec=[uVec u_new];
    end

    gasPerVec=[];
    for i=1:length(uVec(:,1))
        gasPerVec=[gasPerVec trapz(uVec(:,i))/(M-2)];
    end
    z=dz:dz:1-dz;
end
