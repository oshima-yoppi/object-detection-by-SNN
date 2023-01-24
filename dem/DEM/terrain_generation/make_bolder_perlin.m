%(0,-0.5,0,6,3.25,3.25) %(xc,yc,zc) を中心座標、(xr,yr,zr) を半軸の長さ

%    (X-XC)^2     (Y-YC)^2     (Z-ZC)^2
%    --------  +  --------  +  --------  =  1
%      XR^2         YR^2         ZR^2
%% initialize parameter
function f = make_bolder_perlin(k,mode)
% fはモード0:なし、モード1は動画生成、モード2：3次元プロット
    xc = 32;
    yc = 32;
    zc = 0;

    xr = 15+abs(5*randn());
    yr = 15+abs(5*randn());
    zr = 7+abs(5*randn());

    model = zeros(64,64);
    %% Set the range of area to generate the terrain
    x = [0:0.1:6.4];
    y = [0:0.1:6.4];
    %% Generate the random terrain with Perlin noise
    % Initial calculation
    [X, Y] = meshgrid(x,y);
    f = @(t) myinterpolation(t);
    H = perlin_2d(f, 5, X, Y);
    size(H) % 301   301


    %Fractal Perlin noise
    % n = 3;
    % for i = 1:n
    %     xtmp = 2^i*x;
    %     ytmp = 2^i*y;
    %     [Xtmp, Ytmp] = meshgrid(xtmp, ytmp)
    %     Htmp = perlin_2d(f, 10*0.5^i, Xtmp, Ytmp);
    %     H = H + Htmp;
    % end
    %% generate elipsoid
    for i =  1:1:64
        for j = 1:1:64
           if j > yc
               if j < yc + yr*sqrt(1-(i-xc)^2/xr^2)

                    z = genrate_ellipsoid(i,j,xc, yc, zc, xr, yr, zr);
                    model(i,j) = z +H(i,j);
               end
           else
              if j > yc - yr*sqrt(1-(i-xc)^2/xr^2)
                    z = genrate_ellipsoid(i,j,xc, yc, zc, xr, yr, zr);

                    model(i,j) = z+H(i,j) ;
              end
           end
        end
    end
    %% モデル小数点丸め
    model;
    model = round(model,0);
    model;
    %% モード2:3Dプロット
    if mode == 2
        s = surface(model);
        s.EdgeColor = 'none';
        axis equal
        zlim([0 30]);
        xlim([0 64]);
        ylim([0 64]);
        %colorbar
        view(3)
    end
    %% モード1:openV
    if mode==1
        v = VideoWriter('perlin_bolder.avi');
        open(v);
    end
    %% データに保存
    time = 0
    lidar_data = zeros(64,64);
    time_data = zeros(64,64,20);
    for i = 20:-1:0
        time = time+1
        lidar_data(model==i) = 1;
        time_data(:,:,time)=lidar_data;
       %% モード1:動画生成
        if mode ==1
            imagesc(lidar_data);
            colorbar;
            hold on;
            pause(0.1)
            frame = lidar_data;
            writeVideo(v,frame);
        end
        lidar_data =  zeros(64,64); 
    end
    %% モード1:closeV
    if mode ==1
        close(v);
    end
    %% 保存
    filenum = string(k);
    filename ="terrain_generation/perlin_bolder/data_"+filenum;
    save(filename,'time_data')
end

%% 楕円の方程式
function z =genrate_ellipsoid(x ,y, xc, yc, zc, xr, yr, zr)
%x,y
noise = wgn(1,1,-3);
%y = y+noise
z =  zc + zr*sqrt(1-((x-xc)^2/xr^2)-((y-yc)^2/yr^2));

end
%%