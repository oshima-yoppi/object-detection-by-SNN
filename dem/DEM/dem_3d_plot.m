clear
close all
filename = 'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0-3deg)_dem(lidar_noisy)_boulder\model\real_model_5413.mat'
%filename = 'C:\Users\aki\OneDrive - 東京理科大学\研究\64pix_(0deg)_dem(noisy)_evaluate\model\real_model_549.mat'
%filename ='C:\Users\aki\OneDrive - 東京理科大学\研究\real_model_36.mat'

DEM = load(filename,'true_DEM');
DEM = DEM.true_DEM;
% DEM = load(filename,'Lidar_noised_DEM');
% DEM = DEM.Lidar_noised_DEM;

%%Interpolant
sz = size(DEM)
xg = 1:sz(1);
yg = 1:sz(2);
F = griddedInterpolant({xg,yg},double(DEM));

xq = (0:5/15:sz(1))';
yq = (0:5/15:sz(2))';
vq = (F({xq,yq}));
figure(1);
s = surf(vq);

s.EdgeColor = 'none';
xlim([0 200])
ylim([0 200])
zlim([-32 32])
colormap bone
c = colorbar;
view(3)

figure(2)
s = surf(DEM);
s.EdgeColor = 'none';
xlabel('X[m]');
ylabel('Y[m]');
zlabel('Altitude[m]');
xlim([0 64])
ylim([0 64])
zlim([-32 32])
colormap copper
c = colorbar;
c.Label.String = 'Altitude[m]';
view(3)

