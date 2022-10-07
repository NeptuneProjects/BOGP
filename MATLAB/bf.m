clear; clc;
close all;

% Environment parameters
c = 343;       % speed of sound
f = 1000;        % frequency
lambda = c/f;   % wavelength

% ULA-horizontal array configuration
Nsensor = 20;                   % number of sensors
d       = 1/2*lambda;           % intersensor spacing
% d = .1;
q       = (0:1:(Nsensor-1))';   % sensor numbering
xq      = (q-(Nsensor-1)/2)*d;  % sensor locations

SNR = 20; % dB

% number of snapshots
Nsnapshot = 50;

% range of angle space
thetalim = [-90 90];
theta_separation = 0.5;

% Angular search grid
theta = (thetalim(1):theta_separation:thetalim(2))';
Ntheta = length(theta);

% Design/steering matrix (Sensing matrix)
sin_theta = sind(theta);
sensingMatrix = exp(-1i*2*pi/lambda*xq*sin_theta.')/sqrt(Nsensor);

% True DOAs
anglesTrue = [-2; 3; 75];
% anglesTrue = [60];
anglesTracks = repmat(anglesTrue,[1,Nsnapshot]);
sinAnglesTracks = sind(anglesTracks); 

Nsources = numel(anglesTrue);

% received Signal
receivedSignal = zeros(Nsensor,Nsnapshot);
for snapshot = 1:Nsnapshot
    % Fixed strength source amplitude
    source_amp(:,snapshot) = 10 * ones(numel(anglesTrue),1); % equal strength sources
%     source_amp(:,snapshot) = [4; 10]; % fixed strength sources
    Xsource = source_amp(:,snapshot) .* exp(1j*2*pi*rand(numel(anglesTrue),1));
    
    % Complex Gaussian source amplitude
%     source_amp(:,snapshot) = complex(randn(size(anglesTrue)),randn(size(anglesTrue)))/sqrt(2);
%     Xsource = source_amp(:,snapshot);
    
    
    % Represenation matrix (steering matrix)
    transmitMatrix = exp( -1i*2*pi/lambda*xq*sinAnglesTracks(:,snapshot).' )/sqrt(Nsensor);
    
    % Received signal without noise
    receivedSignal(:,snapshot) = sum(transmitMatrix*diag(Xsource),2);
    
    % add noise to the signals
    rnl = 10^(-SNR/20)*norm(Xsource);
    nwhite = complex(randn(Nsensor,1),randn(Nsensor,1))/sqrt(2*Nsensor);
    e = nwhite * rnl;	% error vector
    receivedSignal(:,snapshot) = receivedSignal(:,snapshot) + e;
end

%% Sample Covariance matrix (SCM)
Ryy = receivedSignal * receivedSignal' / Nsnapshot;

figure(100) 
hold on;
%% Conventional beamforming (CBF)
% Function for estimating the spatial spectrum P = a^H R a / a^H a
% This is called the conventional beamformer.
%
% sp - Estimated spatial spectrum
% a - Matrix of steering vectors

na = size(sensingMatrix, 2);
sp = zeros(na, 1);

for i = 1:na
    aa = sensingMatrix(:, i);
    sp(i) = aa'*Ryy*aa/(aa'*aa);
end

% plot(theta, 10*log10(abs(sp)/max(abs(sp))),'displayname', 'CBF')
plot(theta, abs(sp)/max(abs(sp)),'displayname', 'CBF')

%% MVDR (Capon)
% Calculates the spatial spectrum using the minimum variance (Capon's)
% beamformer.
%
% sp - Estimated spatial spectrum
% a - Matrix of steering vectors

sp = zeros(na, 1);

for i = 1:na
    aa = sensingMatrix(:, i);
    sp(i) = 1./(aa'/Ryy*aa);
end

% plot(theta, 10*log10(abs(sp)/max(abs(sp))),'k','displayname', 'MVDR')
plot(theta, abs(sp)/max(abs(sp)),'k','displayname', 'MVDR')

%% MUSIC
% Calculates the spatial spectrum using the MUSIC algorithm.
%
% sp - Estimated spatial spectrum
% a - Matrix of steering vectors
% ns - Number of sources (eigen-values)
ns = Nsources;

[V, D] = eig(Ryy);
Vn = V(:, 1:end-ns);

sp = zeros(na, 1);
for i = 1:na
    aa = sensingMatrix(:, i);
    sp(i) = 1./(aa'*Vn*Vn'*aa);
end

% plot(theta, 10*log10(abs(sp)/max(abs(sp))),'r','displayname', 'MUSIC')
plot(theta, abs(sp)/max(abs(sp)),'r','displayname', 'MUSIC')

%% Eigenvalue method
% Function for estimating the spatial spectrum using the Eigenvalue method.
%
% sp - Estimated spatial spectrum
% a - Matrix of steering vectors
% ns - Number of sources (eigen-values)
ns = Nsources;

[V, D] = eig(Ryy);
d = diag(D);
d = d(1:end-ns);
L = diag(1./d);
Vn = V(:, 1:end-ns);

sp = zeros(na, 1);
for i = 1:na
    aa = sensingMatrix(:, i);
    sp(i) = 1./(aa'*Vn*L*Vn'*aa);
end

% plot(theta, 10*log10(abs(sp)/max(abs(sp))),'g','displayname', 'Eigen')
plot(theta, abs(sp)/max(abs(sp)),'g','displayname', 'Eigen')

%%
hold off;
box on;
% axis([-90 90 -40 2.5])
xlim([-90 90])
xlabel('DOA~[$^\circ$]','interpreter','latex'); ylabel('P~[re max]','interpreter','latex');
set(gca,'fontsize',18)
legend('interpreter','latex','location','northwest')