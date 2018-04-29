data = load('../data/demo_vid.mat');
M = data.M;
hgt = data.vh;
wid = data.vw;
[A, E] = ifralm(M);