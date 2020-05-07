pathImSrc = '\isic\2019\ISIC_2019_Training_Input';
pathImTar = '\isic\2019\official';
%pathImCheck = '\isic\2019\official_check';
fold = dir(pathImSrc);
std_size = [450,600];
preserve_ratio = true;
preserve_size = 600;
crop_black = true;
margin = 0.1;
thresh = 0.3;
resize = true;
use_cc = true;
write_png = false;
write = true;
ind = 1;
all_heights = 0;
all_width = 0;
%initialize
use_cropping = false;
for i=3:length(fold)
    try
       im = imread([pathImSrc '\' fold(i).name]);
    catch
       disp(['Image ' fold(i).name ' failed.'])
       continue
    end
    if crop_black
        lvl = graythresh(rgb2gray(im));
        BW = imbinarize(imgaussfilt(rgb2gray(im),2),lvl*0.2);        
        stats = regionprops('table',BW,'Centroid',...
            'MajorAxisLength','MinorAxisLength');
        if size(stats,1) > 0
            diameters = mean([stats.MajorAxisLength stats.MinorAxisLength],2);
            [diameter_srt,srt_ind] = sort(diameters,'descend');
            %[diameter,ind] = max(diameters);
            radius = diameter_srt(1)/2;
            center = stats.Centroid(srt_ind(1),:);
            % define box
            x_min = int32(center(2)-radius+margin*radius);
            x_max = int32(center(2)+radius-margin*radius);
            y_min = int32(center(1)-radius+margin*radius);
            y_max = int32(center(1)+radius-margin*radius);
            use_cropping = true;
        else
            use_cropping = false;
        end
        if x_min < 1 || x_max > size(im,1) || y_min < 1 || y_max > size(im,2)
            if length(diameter_srt) > 1
                % try second largest
                radius = diameter_srt(2)/2;
                center = stats.Centroid(srt_ind(2),:);
                % define box
                x_min = int32(center(2)-radius+margin*radius);
                x_max = int32(center(2)+radius-margin*radius);
                y_min = int32(center(1)-radius+margin*radius);
                y_max = int32(center(1)+radius-margin*radius);
                if x_min < 1 || x_max > size(im,1) || y_min < 1 || y_max > size(im,2)
                    use_cropping = false;
                end
            else
                use_cropping = false;
            end
        end
        if use_cropping
            mean_inside = mean(im(x_min:x_max,y_min:y_max,:),'all');
            mean_outside = (mean(im(1:x_min,:,:),'all')+mean(im(x_min:x_max,1:y_min,:),'all')+mean(im(x_max:end,:,:),'all')+mean(im(x_min:x_max,y_max:end,:),'all'))/4;
            if mean_outside/mean_inside > thresh
                use_cropping = false;
            end
        end        
        if use_cropping
            %imwrite(im,[pathImCheck '\' fold(i).name]);            
            im = im(x_min:x_max,y_min:y_max,:);
            %imwrite(im,[pathImCheck '\' replace(fold(i).name,'.jpg','_c.jpg')]);
            %disp([fold(i).name ' cropped.'])
        end
    end
    %all_heights(ind) = size(im,1);
    %all_width(ind) = size(im,2);
    %ind = ind+1;
    % resize?
    if resize
        if preserve_ratio
            % long side is resized to target size
            if size(im,1) > size(im,2)
               im = permute(im,[2,1,3]); 
            end
            if size(im,2) ~= preserve_size
                ratio = preserve_size/size(im,2);
                %disp(['Before ' mat2str(size(im))])
                im = imresize(im,[int32(round(size(im,1)*ratio)),preserve_size]);
                %disp(['After ' mat2str(size(im))])
            end
        else
            if size(im,1) > size(im,2)
               im = permute(im,[2,1,3]); 
            end
            if size(im,1) ~= std_size(1) || size(im,2) ~= std_size(2)
               im = imresize(im,std_size); 
            end
        end
    end
    % cc
    if use_cc
        [~,~,~,im_new]=general_cc(double(im),0,6,0);
        im_new = uint8(im_new);
    else
        im_new = im;
    end
    if write
        if write_png
            imwrite(im_new,[pathImTar '\' replace(fold(i).name,'.jpg','.png')]);  
        else
            imwrite(im_new,[pathImTar '\' fold(i).name],'Quality',100);
        end
    end
    if mod(i,1000) == 0
        disp(i)
    end
end