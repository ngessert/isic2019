function out=set_border(in,width,method)
%sets border to either zero method=0,or method=1 to average
if nargin<3
    method=1;
end

temp=ones(size(in));
[y x] = ndgrid(1:size(in,1),1:size(in,2));
temp=temp.*( (x<size(temp,2)-width+1 ) & (x>width) );
temp=temp.*( (y<size(temp,1)-width+1 ) & (y>width) );
out=temp.*in;
if method==1
    out=out+(sum(out(:))./sum(temp(:))) *(ones(size(in))-temp);
end