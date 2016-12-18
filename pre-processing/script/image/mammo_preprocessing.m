%% init
all_file=dir;
all_dir=all_file([all_file(:).isdir]);
all_dir(48)=[];
num_dir = numel(all_dir);
saved_nipple=cell(num_dir,1);
view='MLO';

for i=1:num_dir-2%27
    sub_file=dir(all_dir(i+2).name);
    sub_dir=sub_file([sub_file(:).isdir]);
    num_sub = numel(sub_dir);
    for j=1:num_sub-2
        fixed  = imread(sprintf('./%s/%s/L/%s.png',all_dir(i+2).name,sub_dir(j+2).name,view));
        
        %% left mammography muscle remove
        I=fixed;
        
        %% initial mask
        m =zeros(size(I,1),size(I,2));
        temp_bn=im2bw(I,mean2(I(1:24,1001:1024))/510);
        [temp_bn_idx,temp_bn_label]=bwboundaries(temp_bn,'noholes');
        temp_bn_label_start=reshape(temp_bn_label(1:24,1001:1024),24*24,1); 
        temp_bn_label_start(temp_bn_label_start==0)=nan;
        temp_bn_start=mode(temp_bn_label_start); % find initial points
        if max(temp_bn_idx{temp_bn_start}(:,1))<size(I,1)/3 % adjust init mask when too big mask was selected
            m(1:round(max(temp_bn_idx{temp_bn_start}(:,1))*.7),size(I,2)-round((size(I,2)-min(temp_bn_idx{temp_bn_start}(:,2)))*.7):size(I,2)) = 1; %-- create initial mask
        else
            m(1:round(size(I,1)/3),size(I,2)-round((size(I,2)-min(temp_bn_idx{temp_bn_start}(:,2)))*.7):size(I,2)) = 1; %-- create initial mask
        end
        
        %% segmentation
        seg = region_seg(I, m, 700, .3,1); %-- Run segmentation 
        
        %% cut muscle
        bw_seg=bwboundaries(seg,'noholes'); % get image of segmented muscle area 
        [~,bn_seg_index] = max(cellfun(@(x) numel(x),bw_seg));
        muscle=[max(bw_seg{bn_seg_index}(bw_seg{bn_seg_index}(:,2)==size(I,1),1)), 0 0; size(I,1) min(bw_seg{bn_seg_index}(bw_seg{bn_seg_index}(:,1)==1,2)) size(I,2)]; % compute muscle triangle coordinates
        imshow(I)
        hold on
        fill(muscle(2,:),muscle(1,:),[0,0,0])
        
        %% nipple detection
        bn_im=im2bw(I,1/255);
        bn_index=bwboundaries(bn_im,4);
        [~,mammo_bn_index] = max(cellfun(@(x) numel(x),bn_index));
                bn_index_table=table(bn_index{mammo_bn_index}(:,1),bn_index{mammo_bn_index}(:,2));
        y_intercept=muscle(2,2);
        
        %% find nipple by muscle slope
        inter_num=1;
        intercept_condition=0;
        while intercept_condition==0
            if inter_num>0
                y_intercept=y_intercept-1;
            else
                intercept_condition=1;
                y_intercept=y_intercept+1;
            end
            temp_inter=[];
            line_y=zeros(size(I,1),2);
            for line_x=1:size(I,1)
                line_y(line_x,1)=line_x;
                line_y(line_x,2)=round(line_x*(muscle(2,1)-muscle(2,2))/(muscle(1,1)-muscle(1,2))+y_intercept);
            end
            line_y_table=table(line_y(:,1),line_y(:,2));
            inter_num=numel(find(ismember(line_y_table,bn_index_table)==1));
            nipple=line_y((ismember(line_y_table,bn_index_table)==1),:);
        end
        saveas(gcf, sprintf('./%s_%s_left.jpg',all_file(i+2).name,view));
    end
end

