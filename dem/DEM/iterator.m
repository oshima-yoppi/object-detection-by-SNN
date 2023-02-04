
clear all
close all
clc
pix = 128;
max_angle = 5;
angle = 3;

is_boulder = true;

% addpath(' C:\Users\aki\Documents\GitHub\deep\DEM\terrain_generation');
addpath('terrain_generation');

is_mixangle = 0;
is_noise = 1;
is_evaluate = 0;
evaluate_angle = 0;

% 0deg,5deg�����n�`�ɌX�Ίp���
is_double_terrain = 0;

if is_double_terrain
    folder_name = string(pix)+"pix_("+string(evaluate_angle)+"deg)_dem(noisy)_evaluate_112";
    
elseif is_evaluate %評価用
    folder_name = string(pix)+"pix_("+string(evaluate_angle)+"deg)_dem(noisy)_evaluate_112";


elseif is_noise
    %folder_name = string(pix)+"pix_("+string(angle)+"deg)_dem(noisy)_ver2";
    %folder_name = string(pix)+"pix_("+string(angle)+"deg)_dem(noisy)"; % 9/5
    if is_mixangle; %アングル混在
        folder_name = string(pix)+"pix_(0-"+string(max_angle)+"deg)_dem(noisy)"; % 9/13
    else %アングル固�?
        folder_name = string(pix)+"pix_(0-"+string(angle)+"deg)_dem(lidar_noisy)_boulder"; % 1/17
    end
else
    %folder_name = string(pihx)+"pix_("+string(angle)+"deg)_dem_ver2";
end
mkdir(folder_name)
mkdir(folder_name,'image')
mkdir(folder_name,'label');
mkdir(folder_name,'model');
% 7680:1:16640
for i=0:1:3000
   if is_double_terrain
       evaluate_angle=0
        double_terrain_generation(i,0,pix,evaluate_angle,folder_name,is_noise,is_boulder);
    
   elseif is_evaluate
       evaluate_terrain_generation(i,0,pix,evaluate_angle,folder_name,is_noise,is_boulder);
    
    
   elseif is_mixangle
       angle = round((max_angle)*rand(1)) % スロープ�?(1~5)
       f = fractal_terrain_generation(i,0,pix,angle,folder_name,is_noise,is_boulder);
       
   else
       angle = 3*rand(1)
       f = fractal_terrain_generation(i,0,pix,angle,folder_name,is_noise,is_boulder);
   end
  
%   f = base_DEM(i,2,pix,angle,folder_name,is_noise);

   % 引数�?i:イ�?レーション, mode, pix:解像度, angle:斜度, folder_name:セーブディレクトリ,
   % is_noise:パ�?�リンノイズ付加の有無
   % モー�?0:LOG�?ータ保存�?�モー�?1は動画生�?��?�モー�?2?�?3次�?プロ�?�?
end

function f = count(i)
    f = i^2;
    iter_data =  zeros(1,1); 
    filenum = string(i)
    filename = "crater/data_"+filenum
    save(filename,'iter_data')
    
end