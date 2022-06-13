function Iout= preProcessImg(filename,size)

I = imread(filename);
Iout = imresize(I, size);
%Iout = imresize(I, [414,480]);
%Iout = imresize(I, [414,960]);
end