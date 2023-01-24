%(0,-0.5,0,6,3.25,3.25) %(xc,yc,zc) を中心座標、(xr,yr,zr) を半軸の長さ

%    (X-XC)^2     (Y-YC)^2     (Z-ZC)^2
%    --------  +  --------  +  --------  =  1
%      XR^2         YR^2         ZR^2
%% initialize parameter

function f = make_boulder_perlin(k,mode)
% fはモード0:なし、モード1は動画生成、モード2：3次元プロット
    size_factor=64;
    model = zeros(size_factor,size_factor);
    label_data = zeros(size_factor,size_factor);
    %% 長軸
    xr = 10+abs(3*randn());
    yr = 10+abs(3*randn());
    zr = 6;
    
    %% 中心座標(ボルダー1）
    xc = size_factor/2+20*(-1+(1+1)*rand(1));
    yc = size_factor/2+20*(-1+(1+1)*rand(1));
    zc = 0;
    
    %% 生成したボルダー２が近接した場合に削除する処理
    cnt=0;
    while 1
        xc_2 = size_factor/2+20*(-1+(1+1)*rand(1));
        yc_2 = size_factor/2+20*(-1+(1+1)*rand(1));
        if sqrt(abs(xc-xc_2)^2 + sqrt(yc-yc_2)^2) > 20
            break
        elseif cnt >= 100
            break
        end
        cnt = cnt+1;
    end
    %% Set the range of area to generate the terrain
    x = [0:0.1:6.4];
    y = [0:0.1:6.4];
    %% Generate the random terrain with Perlin noise
    % Initial calculation
    [X, Y] = meshgrid(x,y);
    f = @(t) myinterpolation(t);
    H = perlin_2d(f, 5, X, Y);
    size(H); % 301   301


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
                    label_data(i,j) = 1;

               end
           else
              if j > yc - yr*sqrt(1-(i-xc)^2/xr^2)
                    z = genrate_ellipsoid(i,j,xc, yc, zc, xr, yr, zr);
                    model(i,j) = z+H(i,j) ;
                    label_data(i,j) = 1;

              end
           end
        end
    end
    %% generate elipsoid 2
    for i =  1:1:64
        for j = 1:1:64
           if j > yc_2
               if j < yc_2 + yr*sqrt(1-(i-xc_2)^2/xr^2)

                    z = genrate_ellipsoid(i,j,xc_2, yc_2, zc, xr, yr, zr);
                    model(i,j) = z +H(i,j);
                    label_data(i,j) = 1;
               end
           else
              if j > yc_2 - yr*sqrt(1-(i-xc_2)^2/xr^2)
                    z = genrate_ellipsoid(i,j,xc_2, yc_2, zc, xr, yr, zr);
                    model(i,j) = z+H(i,j) ;
                    label_data(i,j) = 1;
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
        figure(1);
        s = surface(model);
        s.EdgeColor = 'none';
        axis equal
        zlim([0 30]);
        xlim([0 64]);
        ylim([0 64]);
        %colorbar
        view(3)
        figure(2);
        s = surface(label_data);
        view(3)
        %colorbar
        
    end
    %% モード1:openV
    if mode==1
        v = VideoWriter('two_boulder_20step.avi');
        open(v);
    end
    %% データに保存
    time = 0;
    lidar_data = zeros(64,64);
    time_data = zeros(64,64,10);
    max_elevation = max(model(:))
    for i = max_elevation:-1:max_elevation-10
        time = time+1;
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
    if mode ==0
        filenum = string(k);
        filename = "terrain_generation/64pix_two_boulder_label/label_"+filenum
        save(filename,'label_data');
        filename ="terrain_generation/64pix_two_boulder/image_"+filenum
        save(filename,'time_data');
    end
end

%% 楕円の方程式
function z =genrate_ellipsoid(x ,y, xc, yc, zc, xr, yr, zr)
%x,y
noise = wgn(1,1,-3);
%y = y+noise
z =  zc + zr*sqrt(1-((x-xc)^2/xr^2)-((y-yc)^2/yr^2));

end