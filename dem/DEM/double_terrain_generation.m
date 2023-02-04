function f = double_terrain_generation(k,mode,pix,angle,folder_name,is_noise,is_boulder)
    f = mode;
    size_factor = pix;
    time_scale = 20;
    base_0 = zeros(size_factor,size_factor);
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

   crater_num = round(1 + (1 + 6)*rand(1)); 
   center_x_list = zeros(crater_num,1); 
   center_y_list = zeros(crater_num,1); %
   
   alpha = [];
   R = zeros(crater_num,1);

   for crater = 1:1:crater_num
       R(crater) = 3 + (-3 + 15)*rand(1); 
       x_cord = 3 + (3 + pix)*rand(1);
       y_cord = 3 + (3 + pix)*rand(1);
       center_x_list(crater) = round(x_cord);
       center_y_list(crater) = round(y_cord);
   end
   %% �{���_�[

    boulder_num = round(5*rand(1)); 
    boulder_center_x_list =  zeros(boulder_num,1); 
    boulder_center_y_list = zeros(boulder_num,1); 
    boulder_xziku_list = zeros(boulder_num,1);
    boulder_yziku_list = zeros(boulder_num,1); 
    boulder_zziku_list = zeros(boulder_num,1); 
   
   if is_boulder
   for boulder = 1:1:boulder_num
       x_cord = 3 + (3 + pix)*rand(1);
       y_cord = 3 + (3 + pix)*rand(1);
       boulder_center_x_list(boulder) = round(x_cord);
       boulder_center_y_list(boulder) = round(y_cord);
       xr = abs(5*rand(1));
       yr = abs(5*rand(1));
       zr = 3+abs(1*rand(1));
       boulder_xziku_list(boulder) = round(xr);
       boulder_yziku_list(boulder) = round(yr);
       boulder_zziku_list(boulder) = round(zr);
   end

   end
    DEM_0 = put_hazard(base_0,is_noise, center_x_list, center_y_list, R, is_boulder, boulder_center_x_list, boulder_center_y_list, boulder_xziku_list, boulder_yziku_list, boulder_zziku_list);
    DEM_5 = put_hazard(base,is_noise, center_x_list, center_y_list, R, is_boulder, boulder_center_x_list, boulder_center_y_list, boulder_xziku_list, boulder_yziku_list, boulder_zziku_list);
   
    true_DEM = DEM_0;    
    DEM = round(DEM_0,0);

   %% ����
    if mode==2
        figure(1)
        s = surf(DEM);
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

    max_elevation_0 = max(DEM(:));
    min_elevation = max_elevation_0-20;
    for i = max_elevation_0:-1:min_elevation

        time = time+1;
        lidar_data(DEM==i)=1;
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
    if mode == 0
        % 0deg
        mkdir(folder_name,'/0deg');
        curr_folder_name =folder_name+"/0deg";
        mkdir(curr_folder_name,'image')
        mkdir(curr_folder_name,'label');
        mkdir(curr_folder_name,'model');
        filenum = string(k);

        filename = curr_folder_name+"/model/real_model_"+filenum;
        save(filename,'true_DEM');
        
        filename = curr_folder_name+"/model/observed_model_"+filenum;
        save(filename,'DEM');
        
        kyorigazou = mat2gray(true_DEM);
        filename = curr_folder_name+"/model/model_true_"+filenum+'.png';
        imwrite(kyorigazou,filename);
        kyorigazou = mat2gray(DEM);
        filename = curr_folder_name+"/model/model_"+filenum+'.png';
        imwrite(kyorigazou,filename);
        
        filename = curr_folder_name+"/label/label_"+filenum;
        save(filename,'label_data');
     
        filename = curr_folder_name+"/image/image_"+filenum
        save(filename,'time_data');
            
    end
    %% 5deg
    true_DEM = DEM_5;    
    DEM = round(DEM_5,0);
    
    time = 0;
    lidar_data = zeros(size_factor,size_factor);
    label_data = zeros(size_factor,size_factor);
    time_data = zeros(size_factor,size_factor,time_scale);
    
    max_elevation_5 = max(DEM(:));
    min_elevation = max_elevation_5-20;
    for i = max_elevation_5:-1:min_elevation

        time = time+1;
        lidar_data(DEM==i)=1;
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

    if mode == 0
        % 5deg
        mkdir(folder_name,'5deg');
        curr_folder_name =folder_name+"/5deg";
        mkdir(curr_folder_name,'image')
        mkdir(curr_folder_name,'label');
        mkdir(curr_folder_name,'model');
        filenum = string(k);

        filename = curr_folder_name+"/model/real_model_"+filenum;
        save(filename,'true_DEM');
        
        filename = curr_folder_name+"/model/observed_model_"+filenum;
        save(filename,'DEM');
        
        kyorigazou = mat2gray(true_DEM);
        filename = curr_folder_name+"/model/model_true_"+filenum+'.png';
        imwrite(kyorigazou,filename);
        kyorigazou = mat2gray(DEM);
        filename = curr_folder_name+"/model/model_"+filenum+'.png';
        imwrite(kyorigazou,filename);

        filename = curr_folder_name+"/label/label_"+filenum;
        save(filename,'label_data');
     

        filename = curr_folder_name+"/image/image_"+filenum
        save(filename,'time_data');
    end