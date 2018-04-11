obj = VideoReader('IMG_6943.m4v');
frames = [];

while hasFrame(obj)
  this_frame = readFrame(obj);
  frames{end + 1} = this_frame;
%   thisfig = figure();
%   thisax = axes('Parent', thisfig);
%   image(this_frame, 'Parent', thisax);
%   title(thisax, sprintf('Frame #%d', k));
end
