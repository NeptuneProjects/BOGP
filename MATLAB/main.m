close all;
clear all;

load zreplica_250.mat;

% source location
sr = 3e3;   % source range (m)
sd = 40;    % source depth (m)

% received field
r = squeeze(p1(find(rd==sd),find(rr==sr),:));
K = r*r';

% MFP
for mm = 1:length(rr)
    for nn = 1:length(rd)
        svectmp = squeeze(p1(nn,mm,:));
        wbart = svectmp/norm(svectmp);
        
        asurf(mm,nn) = abs(wbart'*K*wbart);
    end
end
nasurf = asurf/max(max(asurf));

figure(1)
pcolor(rr,rd,10*log10(nasurf.')); shading flat; hold on;
axis ij;
caxis([-10 0]);
xlabel('Range (m)');
ylabel('Depth (m)');
colorbar;

