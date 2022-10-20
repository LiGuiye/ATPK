% clc;
% clear;
function main_function_ATPK_Guiye(scale, data_type, slice, slice_count)
Sill_min=1;
Range_min=0.5;
L_sill=20;
L_range=20;
rate=0.1;
W=1; % 2 is better than 1, also time consuming
sigma=scale/2;
PSF=PSF_template(scale,W,sigma);%%%Gaussian PSF

% dataset config
channel = 2;
start_size = 8;
max_lag = start_size / 2;
if data_type == "Solar"
%   img_path = "/home/guiyli/Documents/DataSet/Solar/npyFiles/dni_dhi/2014/";
    img_path = "/lustre/scratch/guiyli/Dataset_NSRDB/DIP/Solar2014_removed/";
    real_size = 256;
else
%     img_path = "/home/guiyli/Documents/DataSet/Wind/2014/removed_u_v/";
    img_path = "/lustre/scratch/guiyli/Dataset_WIND/DIP/Wind2014_removed/u_v/";
    real_size = 512;
end

image_list = dir(img_path+'*.npy');
step = fix(length(image_list) / slice_count);
% slice: 0-19
start = slice * step + 1;
if slice == slice_count-1
    stop = length(image_list);
else
    stop = (slice + 1) * step;
end
fprintf("Data type:%s\n",data_type);
fprintf("Scale:%d\n",scale);
fprintf("Start:%d Stop:%d\n",start,stop);

for n=start:stop
    curr_path = img_path + image_list(n).name;
    real = readNPY(curr_path);
    real = imresize(real,[real_size,real_size],'nearest');
    if ~isa(real,'double')
        real = double(real);
    end
    hr = imresize(real, [start_size*scale,start_size*scale], 'box');
    lr = imresize(real, [start_size,start_size], 'box');
    for c=1:channel
        output = ATPK_Guiye(lr(:,:,c),scale,Sill_min,Range_min,L_sill,L_range,rate,max_lag,W,PSF);
        if data_type == "Solar"
            save_folder_hr = strrep(img_path,"Solar2014_removed","ATP_hr_scale"+string(scale));
            save_folder_fake = strrep(img_path,"Solar2014_removed","ATP_fake_scale"+string(scale));
        else
            save_folder_hr = strrep(img_path,"Wind2014_removed","ATP_hr_scale"+string(scale));
            save_folder_fake = strrep(img_path,"Wind2014_removed","ATP_fake_scale"+string(scale));
        end

        if ~exist(save_folder_hr, 'dir')
            mkdir(save_folder_hr);
        end
        if ~exist(save_folder_fake, 'dir')
            mkdir(save_folder_fake);
        end

        writeNPY(output, save_folder_fake+"/"+extractBefore(image_list(n).name,".npy")+"_channel"+string(c)+'.npy');
        writeNPY(hr(:,:,c), save_folder_hr+"/"+extractBefore(image_list(n).name,".npy")+"_channel"+string(c)+'.npy');
    end
end

fprintf("Done");

% figure, image(lr(:,:,2))
% figure, image(hr(:,:,2))
% figure, image(output)