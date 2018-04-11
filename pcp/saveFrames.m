function saveFrames(frames, dir)
    j = 1;
    skip = 10;
    for i = 1 : skip :length(frames)
        I = rgb2gray(frames{i});
        imwrite(I, strcat(dir, num2str(j),'.jpg'));
        j = j + 1;
    end
end