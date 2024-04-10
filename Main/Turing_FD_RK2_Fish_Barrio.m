% floorDiv() required Fixed-Point Designer
% show_patterns was not used
clear all; clc; close all;
% 3.)
a = 0.45 * 6;
b = 6;
alp = 0.899;
bet = -0.91;
gam = -alp;
r2 = 2;
r3 = 3.5/1;


Nx = 128; % Silahkan ukuran matriks ini diubah 2^n (128,256,512,1024)
Ny = Nx;

dx = 1; % space step
dy = 1;

x1 = (0:(Nx-1))*dx;
y1 = (0:(Ny-1))*dy;

x= zeros(Nx,Ny);
y= zeros(Nx,Ny);


for j=1:Ny
  x(:,j) = x1(:);
end

for i=1:Nx
    y(i,:) =y1(:);
end

%  dx = 2 / (Nx-1); % Steps x axis
%  dy = 2 / (Ny-1); % Steps y axis
% [x,y] = meshgrid(-1:dx:1, -1:dy:1);

T = 0.1; % total time
dt = 0.01; % time step
% n = floor(T/dt); % number of iterations

 

% U = 1*rand(Nx,Ny)-0.50;
% V = 1*rand(Nx,Ny)-0.50;
% U = randi([0, 1], [Nx,Ny]);
% V = randi([0, 1], [Nx,Ny]);
U  = dlmread('u.txt');
U = reshape(U,Nx,Ny);
U = U.';
V  = dlmread('v.txt');
V = reshape(V,Nx,Ny);
V =V.';


Ud = U;
Vd = V;
%  return
% Initialiaze K1 and K2
K1U = zeros(Nx,Ny);
K1V = zeros(Nx,Ny);
K2U = zeros(Nx,Ny);
K2V = zeros(Nx,Ny);
% return
% 8.)
% fig, axes = plot(3,3);
% step_plot = floor(n/1);
 step_plot = 1000;
% We simulate the PDE with the finite difference method.
t = 0;
for i = 1:30000
    
    [K1U,K1V] = RK2(U,V,a,b,dx,Nx,Ny,alp,bet,gam,r2,r3);
    Ud = U + dt *K1U;
    Vd = V + dt *K1V;
 
    % Neumann conditions: derivatives at edges are null.
    [Ud,Vd] = neumann(Ud,Vd,Nx,Ny);

    
   [K2U,K2V] = RK2(Ud,Vd,a,b,dx,Nx,Ny,alp,bet,gam,r2,r3);
    U = U + (dt/2) * (K1U+K2U);
    V = V + (dt/2) * (K1V+K2V);
%    
%        % Neumann conditions: derivatives at edges are null.
     [U,V] = neumann(U,V,Nx,Ny);
    t = t +dt;
    
 
    % We plot the state of the system at 9 different times.
    if mod(i,step_plot)==0 %&& i < 9 * step_plot
        minU = min(U(:));  maxU = max(U(:));
         fprintf(2, 'i = %5d ; time = %7.4f ; min U = %7.4f ; max U = %7.4f\n', i, t, minU, maxU);
        
%         figure(i)
        figure(gcf)
        colormap jet;
        surf(x,y,U);  shading interp;  view(3); pause(1e-5); zlim([-0.55 1.25]);
    end
end


function [KU,KV] = RK2(U,V,a,b,dx,Nx,Ny,alp,bet,gam,r2,r3)
KU = zeros(Nx,Ny);
KV = zeros(Nx,Ny);
deltaU = zeros(Nx,Ny);
deltaV = zeros(Nx,Ny);

    % We compute the Laplacian of u and v.
    deltaU = laplacian(U,dx,Nx,Ny);
    deltaV = laplacian(V,dx,Nx,Ny);
    % We look for K1 & K2 value
    KU = (a * deltaU + alp * U + V - r2 * U.*V - alp * r3 * U.*(V.^2));
    KV = (b * deltaV + gam * U + bet * V + r2 * U.*V + alp * r3 * U.*(V.^2));
end

function [U,V] = neumann(U,V,Nx,Ny)
    U(1,:)  = U(2,:);
    U(Nx,:) = U (Nx-1,:);
    U(:,1)  = U (:,2);
    U(:,Ny) = U (:,Ny-1);
    V(1,:)  = V(2,:);
    V(Nx,:) = V(Nx-1,:);
    V(:,1)  = V (:,2);
    V(:,Ny) = V (:,Ny-1);
end

function L = laplacian(Z,h,Nx,Ny)
L = Z;
 for j=2:Ny-1
    for i = 2:Nx-1
        L(i,j) = (Z(i+1,j)-2*Z(i,j)+Z(i-1,j))/h^2 + (Z(i,j+1)-2*Z(i,j)+Z(i,j-1))/h^2;
    end
 end
end