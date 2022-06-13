img = imread('D:\gray_orsize.jpg');  %// get a digit
dx = -1+2*rand(size(img)); 
dy = -1+2*rand(size(img)); 
sig=4; 
alpha=60;
H=fspecial('gauss',[7 7], sig);
fdx=imfilter(dx,H);
fdy=imfilter(dy,H);
n=sum((fdx(:).^2+fdy(:).^2)); %// norm (?) not quite sure about this "norm"
fdx=alpha*10*fdx./n;
fdy=alpha*10*fdy./n;
[y x]=ndgrid(1:size(img,1),1:size(img,2));
figure;
imagesc(img); colormap gray; axis image; axis tight;
hold on;
quiver(x,y,fdx,fdy,0,'r');
new = griddata(x-fdx,y-fdy,double(img),x,y);
new(isnan(new))=0;
figure;
subplot(121); imagesc(img); axis image;
subplot(122); imagesc(new); axis image;
colormap gray
