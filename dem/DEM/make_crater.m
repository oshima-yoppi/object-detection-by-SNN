function f = make_crater(k,mode)
   % fはモード0:なし、モード1は動画生成、モード2：3次元プロット
    f = mode;
    size_factor = 64;
    center = size_factor/2;
    model = zeros(size_factor,size_factor);


    R = 15 + abs(5*randn(1));
   % H_r = 150 + abs(5*randn())
    H_ro = 0.036*(2*R)^1.014;
    H_r = H_ro;
    H_c = 0.196*(2*R)^1.010 - H_ro; 
    W_r = 0.257*(2*R)^1.011;
    % RANGE

    alpha = (H_c+H_r)*R/(H_c+H_ro);
    beta = R+(1-(H_c+H_r)/(H_c+H_ro))*W_r;

    A = -3*R^3 + 2*R^2*beta + 2*R*beta^2 + 2*beta^3;

    for i =  1:1:size_factor
        for j = 1:1:size_factor

            y = 0;
            r = sqrt(abs(i-center)^2 + abs(j-center)^2)+y;
             if r <= alpha
                 h = (H_c+H_ro)*(r^2/R^2)-H_c;
             elseif r <= R
                 h = ((H_c + H_ro)^2/(H_r - H_ro) - H_c)*((r/R) - 1)^2 + H_r;
             elseif r < beta
                 h =  (H_r*(R+W_r)^3*A)/(W_r*beta^4*(R-beta)^2*(3*R^2+3*R*W_r+W_r^2)) * (r-R)^2*(r-beta*(1+(beta^3-R^3)/A))+H_r;
            else    
                h = H_r*(R+W_r)^3/((R+W_r)^3-R^3)*(r/R)^(-3) - (H_r*R^3)/((R+W_r)^3-R^3);
             end
           
            model(i,j) = h;
        end
    end
    model;
    true_DEM = model;
    DEM = round(model,0);

    a = max(model(:)); %22013
    b = a-10;
%% 
    if mode ==2
        s = surface(model);
        s.EdgeColor = 'none';
        zlim([-50 50])
        colorbar
        view(3)
    end
%   
    if mode==1
        v = VideoWriter('4_22_20step.avi');
        open(v);
    end
%%

    time = 0;
    lidar_data = zeros(size_factor,size_factor);
    time_data = zeros(size_factor,size_factor,10); % 128 128 20 

    for i =  a:-1:b
        time = time + 1;
        lidar_data(model==i) = 1;
        time_data(:,:,time) = lidar_data;
%%
        if mode ==1
            imagesc(lidar_data);
            colorbar;
            hold on;
            pause(0.1)
            frame = lidar_data;
            writeVideo(v,frame);
        end
%%
        lidar_data =  zeros(size_factor,size_factor);  
    end
    if mode ==1
        close(v);
    end
    filenum = string(k)
    filename = "simple_crater/image/data_"+filenum;
    save(filename,'time_data');
    
    filename = "simple_crater/model(t-10)/observed_model_"+filenum
    save(filename,'DEM');
    kyorigazou = mat2gray(DEM);
    filename = "simple_crater/model(t-10)/observed_model_"+filenum+'.png';
    imwrite(kyorigazou,filename);
    
    filename = "simple_crater/model/observed_model_"+filenum
    save(filename, 'model');
    kyorigazou = mat2gray(model);
    filename = "simple_crater/model/observed_model_"+filenum+'.png';
    imwrite(kyorigazou,filename);

end



