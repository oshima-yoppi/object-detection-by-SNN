function f = fractal_terrain_generation(k,mode,pix,angle,folder_name,is_noise,is_boulder)
    f = mode;

%% 斜面 & フラクタルの付与
    size_factor = pix;
    time_scale = 10;
    base = zeros(size_factor,size_factor);
    label_data = zeros(size_factor,size_factor);
    time_data = zeros(size_factor,size_factor,time_scale);
    
    direct = round(rand(1),0);
    up_down= round(rand(1),0);
    
    if up_down == 1
        up_down = 1;
    else 
        up_down = -1;
    end

    for i=1:1:size_factor
        for j=1:1:size_factor
            if direct==1
                base(:,j) = (up_down)*j*tan(deg2rad(angle));
            else
                base(i,:) = (up_down)*i*tan(deg2rad(angle));   
            end
        end
    end

   %% クレータの個数 & 座標を決定
   crater_num = round(1 + (1 + 6)*rand(1)); % クレータ個数(1~6)
   center_x_list = zeros(crater_num,1); %　クレータ中心座標_x
   center_y_list = zeros(crater_num,1); % クレータ中心座標_y
   
   alpha = [];
   R = zeros(crater_num,1);

   for crater = 1:1:crater_num
       R(crater) = 3 + (-3 + 15)*rand(1); %クレータ半径(3~10)
       x_cord = 3 + (3 + pix)*rand(1);
       y_cord = 3 + (3 + pix)*rand(1);
       center_x_list(crater) = round(x_cord);
       center_y_list(crater) = round(y_cord);
   end
   %% ボルダーの個数 & 座標を決定

    boulder_num = round(15*rand(1)); % ボルダー個数(0~5)
    boulder_center_x_list =  zeros(boulder_num,1); % ボルダー中心座標x
    boulder_center_y_list = zeros(boulder_num,1); % ボルダー中心座標y
    boulder_xziku_list = zeros(boulder_num,1); % ボルダーx軸長さ
    boulder_yziku_list = zeros(boulder_num,1); % ボルダーy軸長さ
    boulder_zziku_list = zeros(boulder_num,1); % ボルダーz軸長さ
   
   if is_boulder
   for boulder = 1:1:boulder_num
       % 座標
       x_cord = 3 + (3 + pix)*rand(1);
       y_cord = 3 + (3 + pix)*rand(1);
       boulder_center_x_list(boulder) = round(x_cord);
       boulder_center_y_list(boulder) = round(y_cord);
       % 大きさ
       xr = abs(5*rand(1));
       yr = xr;
       zr = 3+abs(1*rand(1));
       boulder_xziku_list(boulder) = round(xr);
       boulder_yziku_list(boulder) = round(yr);
       boulder_zziku_list(boulder) = round(zr);
   end

   end
 %% パーリンノイズの度合い
    noise_min = 0.2;
    noise_max = 0.8;
    noise_val = (noise_max-noise_min).*rand(1)+noise_min;
    
 %% put_hazard
    DEM = put_hazard(base,is_noise, center_x_list, center_y_list, R, is_boulder, boulder_center_x_list, boulder_center_y_list, boulder_xziku_list, boulder_yziku_list, boulder_zziku_list);
   
 %% true_DEM
    true_DEM = DEM;    
 %% 
    Lidar_noised_DEM = true_DEM + 0.1*randn(pix);
    DEM = round(Lidar_noised_DEM,0);

    %% 動画 v open

    if mode == 1
        v = VideoWriter('SLOPE_two_craters_image.avi')
        open(v)
    end
   %% 三次元プロット
    if mode==2
        figure(1)
        s = surf(true_DEM);
        s.EdgeColor = 'none';
        xlabel('X');
        ylabel('Y');
        zlim([-20 20])
        colormap gray
        colorbar
        view(3)
        savefig('model.fig')
        figure(2)
        s = surf(Lidar_noised_DEM);
        s.EdgeColor = 'none';
        xlabel('X');
        ylabel('Y');
        zlim([-20 20])
        colormap gray
        colorbar
        view(3)
        savefig('model.fig')
       

        
    end
    time = 0;
    lidar_data = zeros(size_factor,size_factor);
    
    %% LIDAR 観測パルス 生成
    max_elevation = max(Lidar_noised_DEM(:));
    min_elevation = max_elevation-10;
    for i = max_elevation:-1:min_elevation

        time = time+1;
        lidar_data(Lidar_noised_DEM==i)=1;
        time_data(:,:,time) = lidar_data;
        if mode == 1
            imagesc(lidar_data);
            colorbar;
            hold on;
            pause(0.1);
            frame = lidar_data;
            writeVideo(v,frame);
        end
        lidar_data = zeros(size_factor,size_factor);
    end
    if mode == 1
        close(v);
    end
    


    %% 教師データとして保存
    if mode == 0
            
        filenum = string(k);
        % true_DEM.mat hazard_label評価用
        filename = folder_name+"/model/real_model_"+filenum;
        save(filename,'true_DEM');
        
        % Lidar_noised_DEM.mat 注意:量子化はしていない
        filename = folder_name+"/model/Lidar_noised_model_"+filenum;
        save(filename,'Lidar_noised_DEM');
        
        % observed_DEM.mat hazard_label評価用 8/29追加
        filename = folder_name+"/model/observed_model_"+filenum;
        save(filename,'DEM');
        
        % model_png 見た目で評価（丸め、なしとあり）
        kyorigazou = mat2gray(true_DEM);
        filename = folder_name+"/model/model_true"+filenum+'.png';
        imwrite(kyorigazou,filename);
        kyorigazou = mat2gray(Lidar_noised_DEM);
        filename = folder_name+"/model/model_Lidar_noised"+filenum+'.png';
        imwrite(kyorigazou,filename);
        
        % label_data.mat crater中心を塗りつぶした
        %filename = folder_name+"/label/label_"+filenum;
        %save(filename,'label_data');
     
        % time_data lidar再現データ
        filename = folder_name+"/image/image_"+filenum
        save(filename,'time_data');
    end