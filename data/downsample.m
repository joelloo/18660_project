frames = dir('./frames/traffic/*.jpg');
for i = 1 : length(frames)
    img = imread(strcat('./frames/traffic/',frames(i).name));
    img = imresize(img, 0.125);
    imwrite(img, strcat('./frames/traffic_downsampled/', frames(i).name));
end