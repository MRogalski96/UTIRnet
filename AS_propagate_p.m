function uout = AS_propagate_p(uin, z, n0, lambda, dx)
% Angular spectrum propagation

[Ny,Nx] = size(uin);
k = 2*pi/lambda; 

dfx = 1/Nx/dx; fx = -Nx/2*dfx : dfx : (Nx/2-1)*dfx; 
dfy = 1/Ny/dx; fy = -Ny/2*dfy : dfy : (Ny/2-1)*dfy; 

if  z<0 
    p = fftshift(k*z*sqrt(n0^2 - lambda^2*(ones(Ny,1)*(fx.^2)+(fy'.^2)*ones(1,Nx))));
    p = p - p(1,1);
    kernel = exp(-1i*p);
    ftu = kernel.*fft2(conj(uin));
    uout = conj(ifft2(ftu));
else
    p = fftshift(k*z*sqrt(n0^2 - lambda^2*(ones(Ny,1)*(fx.^2)+(fy'.^2)*ones(1,Nx))));
    p = p - p(1,1);
    kernel = exp(1i*p);
    ftu = kernel.*fft2(uin);
    uout = ifft2(ftu);
end
end

