function [Rw,Gw,Bw]=NormDerivative(in, sigma, order)

if(nargin<3) order=1; end

R=in(:,:,1);
G=in(:,:,2);
B=in(:,:,3);

if(order==1)
    Rx=gDer(R,sigma,1,0);
    Ry=gDer(R,sigma,0,1);
    Rw=sqrt(Rx.^2+Ry.^2);
    
    Gx=gDer(G,sigma,1,0);
    Gy=gDer(G,sigma,0,1);
    Gw=sqrt(Gx.^2+Gy.^2);
    
    Bx=gDer(B,sigma,1,0);
    By=gDer(B,sigma,0,1);
    Bw=sqrt(Bx.^2+By.^2);
end

if(order==2)        %computes frobius norm
    Rxx=gDer(R,sigma,2,0);
    Ryy=gDer(R,sigma,0,2);
    Rxy=gDer(R,sigma,1,1);
    Rw=sqrt(Rxx.^2+4*Rxy.^2+Ryy.^2);
    
    Gxx=gDer(G,sigma,2,0);
    Gyy=gDer(G,sigma,0,2);
    Gxy=gDer(G,sigma,1,1);
    Gw=sqrt(Gxx.^2+4*Gxy.^2+Gyy.^2);
    
    Bxx=gDer(B,sigma,2,0);
    Byy=gDer(B,sigma,0,2);
    Bxy=gDer(B,sigma,1,1);
    Bw=sqrt(Bxx.^2+4*Bxy.^2+Byy.^2);
end