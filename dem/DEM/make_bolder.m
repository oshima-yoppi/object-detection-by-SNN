function f = make_crater(k,mode)
    % fはモード0:なし、モード1は動画生成、モード2：3次元プロット
    f = mode
    size_factor = 64
    center = size_factor/2
    model = zeros(size_factor,size_factor);


    R = 15 + 5*rand(1)
    H_r = 10
    H_ro = 0.036*(2*R)^1.014
    H_r = H_ro
    H_c = 0.196*(2*R)^1.010 - H_ro 
    W_r = 0.257*(2*R)^1.011
    % RANGE

    alpha = (H_c+H_r)*R/(H_c+H_ro)
    beta = R+(1-(H_c+H_r)/(H_c+H_ro))*W_r

    A = -3*R^3 + 2*R^2*beta + 2*R*beta^2 + 2*beta^3

    for i =  1:1:size_factor
        for j = 1:1:size_factor


            r = sqrt(abs(i-center)^2 + abs(j-center)^2);
             if r <= alpha
                 h = (H_c+H_ro)*(r^2/R^2)-H_c;
             elseif r <= R
                 h = ((H_c + H_ro)^2/(H_r - H_ro) - H_c)*((r/R) - 1)^2 + H_r;
             elseif r < beta
                 h =  (H_r*(R+W_r)^3*A)/(W_r*beta^4*(R-beta)^2*(3*R^2+3*R*W_r+W_r^2)) * (r-R)^2*(r-beta*(1+(beta^3-R^3)/A))+H_r;
            else    
                h = H_r*(R+W_r)^3/((R+W_r)^3-R^3)*(r/R)^(-3) - (H_r*R^3)/((R+W_r)^3-R^3);
             end
            y = wgn(1,1,-3);
            model(i,j) = -(h+y);
        end
    end
    model;
    model = round(model,0);
    model;
    % model = model.*10000
    % model
    a = max(model(:)) %22013
    b = min(model(:)) %5885
%% 
    if mode ==2
        s = surface(model);
        s.EdgeColor = 'none';
        zlim([-50 50])
        %colorbar
        view(3)
    end
%   
    if mode==1
        v = VideoWriter('peaks_100_noise.avi');
        open(v);
    end
%%

    time = 0;
    lidar_data = zeros(size_factor,size_factor);
    time_data = zeros(size_factor,size_factor,20); % 128 128 20 

    for i =  10:-1:-9
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
    filename = "bolder/data_"+filenum
    save(filename,'time_data')
    

end
