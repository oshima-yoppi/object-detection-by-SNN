
fid  = fopen('DTM_MAP_02_N00E000S03E003SC.img','r');
data = fread(fid,[12288,12288],'short','l');
fclose(fid);
% surf(data);
% colormap(jet);
size(data)
% imagesc(data);colorbar;
% a = colormap('jet');

trim = data(1:500,1:500);
a = max(trim(:)) %22013
b = min(trim(:)) %5885
size(trim)
% subplot(1,2,1), imagesc(data);colorbar;
% subplot(1,2,2), imagesc(trim);colorbar;
% imagesc(trim);colorbar;
% a = colormap('jet');
lidar_data = zeros(500,500);
% image(lidar_data)
% colorbar
% for i = 1:1:50
%     lidar_data(i,i)=255;
%     imagesc(lidar_data);
%     colorbar;
%     pause(0.1)
% end
% lidar_data(1,1)= 255;
% lidar_data(1,1)
% imagesc(lidar_data)
% colorbar
v = VideoWriter('peaks.avi');
open(v);
size(trim)
for i =  a:-1:b
%     disp(i)
    lidar_data(trim==i) = 1;
    
    if rem(i,100) == 0
        imagesc(lidar_data);
        colorbar;
        hold on;
        pause(0.01)
        frame = lidar_data;
        writeVideo(v,frame);
        lidar_data =  zeros(500,500);
    end
    

    
end
close(v);
imagesc(lidar_data);
colorbar;

% f1 = figure;
% f2 = figure;
% figure(f1);
% imagesc(data);colorbar;
% figure(f2);
% imagesc(trim);colorbar;


s = surface(trim);
s.EdgeColor = 'none';
%colorbar
view(3)