fixed  = imread(sprintf('./%s/%s/L/%s.png',all_dir(i+2).name,sub_dir(j+2).name,view));
moving = imread(sprintf('./%s/%s/R/%s.png',all_dir(i+2).name,sub_dir(j+2).name,view));
[optimizer, metric] = imregconfig('monomodal')
movingRegistered = imregister(moving, fixed, 'affine', optimizer, metric);
