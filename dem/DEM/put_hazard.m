function DEM = put_hazard(base,is_noise, center_of_x, center_of_y,R, is_boulder, boulder_center_x_list, boulder_center_y_list, boulder_xziku_list, boulder_yziku_list, boulder_zziku_list)
    
    size_factor = size(base);
    DEM = zeros(size_factor);
 
    if is_noise
    x = [0:0.1:64];
    y = [0:0.1:64];
    [X, Y] = meshgrid(x,y);
    f = @(t) myinterpolation(t);
    H = perlin_2d(f, 1, X, Y);
    size(H);
    noise_min = 0.2;
    noise_max = 0.8;
    noise_val = (noise_max-noise_min).*rand(1)+noise_min;
    else
        H = zeros(size_factor);
        noise_val = 0;
    end

    
   %% 障害物付与
    for i =  1:1:size_factor(1)
        for j = 1:1:size_factor(2)
               for ind = 1:1:length(center_of_x)
                    x = center_of_x(ind);
                    y = center_of_y(ind);
                    r = sqrt(abs(i-x)^2 + abs(j-y)^2);
                    h = base(i,j);
                    
                    H_ro = 0.036*(2*R(ind))^1.014;
                    H_r = H_ro;
                    H_c = 0.196*(2*R(ind))^1.010 - H_ro ;
                    W_r = 0.257*(2*R(ind))^1.011;
                    alpha = (H_c+H_r)*R(ind)/(H_c+H_ro); %クレータ内縁
                  %%  
                    if r <= alpha
                        h = (H_c+H_ro)*(r^2/R(ind)^2)-H_c;
                        label_data(i,j) = 1;
                    else
                        h = H_r*(R(ind)+W_r)^3/((R(ind)+W_r)^3-R(ind)^3)*(r/R(ind))^(-3) - (H_r*R(ind)^3)/((R(ind)+W_r)^3-R(ind)^3);
                    end
                   base(i,j) = base(i,j) + h;
               end
               if is_boulder
                   for ind = 1:1:length(boulder_center_x_list)
                       xc = boulder_center_x_list(ind);
                       yc = boulder_center_y_list(ind);
                       zc = 0;
                       xr = boulder_xziku_list(ind);
                       yr = boulder_yziku_list(ind);
                       zr = boulder_zziku_list(ind);
                       h = 0;
                   %% ボルダー 追加
                        if j > yc
                            if j < yc + yr*sqrt(1-(i-xc)^2/xr^2)
                                z = generate_ellipsoid(i,j,xc, yc, zc, xr, yr, zr);
                                h = z + H(i,j);
                            end
                        else
                            if j > yc - yr*sqrt(1-(i-xc)^2/xr^2)
                                z = generate_ellipsoid(i,j,xc, yc, zc, xr, yr, zr);
                                h = z + H(i,j) ;
                            end
                        end
                        base(i,j) = base(i,j) + h;
                   end
               end
               DEM(i,j) = base(i,j) + noise_val * H(i,j);

        end
    end





end
%% 楕円の方程式
function z =generate_ellipsoid(x ,y, xc, yc, zc, xr, yr, zr)
%x,y
%noise = wgn(1,1,-3);
%y = y+noise
z =  zc + zr*sqrt(1-((x-xc)^2/xr^2)-((y-yc)^2/yr^2));

end