# -*- coding: utf-8 -*-
using Distributions, DistributionsAD, LinearAlgebra, KernelFunctions, BlockDiagonals, Zygote, LinearAlgebra, Flux
import Base.+, Base.-, Base.*




struct SEKernel <: KernelFunctions.Kernel
     
    se_variance
    se_lengthscale

end

Flux.@functor SEKernel


SEKernel() = SEKernel(zeros(1,1), zeros(1,1))


function KernelFunctions.kernelmatrix(m::SEKernel,x::Matrix,y::Matrix)

    diffed = sum((Flux.unsqueeze(x,3) .- Flux.unsqueeze(y,2)).^2,dims=1)[1,:,:]
    
    return exp(m.se_variance[1,1]) .* exp.(- 0.5 *exp(m.se_lengthscale[1,1]) .* diffed)

end

KernelFunctions.kernelmatrix(m::SEKernel,x::Matrix) = KernelFunctions.kernelmatrix(m,x,x)




function logdetcholesky(X)
    
    m,n = size(X)
    
    return 2 * sum(log.(diag(cholesky(Symmetric(X.+Diagonal(ones(n).*1e-5))).L)))

end




struct SVGP
    
    I
    m
    L
    k


end
Flux.@functor SVGP

function SVGP(n_ind,n_dims=1) 
   
    SVGP(rand(n_dims,n_ind).*4 .-2, zeros(1,n_ind), randn(n_ind,n_ind).*0.1,SEKernel(zeros(1,1), - 2 .*ones(1,1)))

end


function SVGP(inducing_points::Matrix{Float64}) 
    n_ind = size(inducing_points,2)
    SVGP(inducing_points, zeros(1,n_ind), randn(n_ind,n_ind).*0.1,SEKernel(zeros(1,1), - 2 .*ones(1,1)))

end



function (m::SVGP)(x)
    
    kmm = kernelmatrix(m.k, m.I)
    kmmi = inv(kmm .+ Diagonal(ones(size(m.I,2)).*1e-4))
    knn = kernelmatrix(m.k, x)
    kmn = kernelmatrix(m.k, m.I, x)
    
    S = m.L*transpose(m.L)
    
    
    return m.m*kmmi*kmn, knn .- transpose(kmn)*kmmi*(kmm .- S)*kmmi*kmn

end



struct SFGP
    
    gp
    lower
    upper
    vals
    s
    
    function SFGP(gp::SVGP,n_points=10,limit=3)
    
        lower = vcat(-1e10,collect(range(-limit,limit,length=n_points-1)))[:,:]
        upper = vcat(collect(range(-limit,limit,length=n_points-1)),1e10)[:,:]

        #vals = randn(n_points,1)
        vals = (collect(range(-limit,limit,length=n_points)))
    
    return new(gp,lower,upper,vals,ones(1,1))

end

end
Flux.@functor SFGP

Flux.params(m::SFGP) = Flux.params(m.gp,m.vals)

getProbs(m,s,l,u) = cdf.(Normal(m,s),u) .- cdf.(Normal(m,s),l)

Base.broadcastable(m::SFGP) = (m,)


lpdf_norm(m,s,y) = -0.5 * log(2*3.14*s^2) - 0.5/(s^2)*(m-y)^2


function lpdf(mm::SFGP,x,y)
    m,S = mm.gp(x) 
    s = sqrt.(diag(S))
      
    #return sum(probs .* logpdf.(Normal.(sigvals,abs(mm.s[1,1])),y))
    sig = abs(mm.s[1,1])
    return mean(lpdf_norm.(m[:], sig,y[:]) .+ 0.5 *s/sig^2)
end


function kldiv(m1,S1,m2,S2)
    _,N = size(m1)
        
    mdiff = m2.-m1
    
    S2i = inv(S2.+Diagonal(ones(N).*1e-4))
        
    return 0.5 * (logdetcholesky(S2)-logdetcholesky(S1) - N + tr(S2i*S1) + sum(mdiff*S2i*transpose(mdiff)))

end


getzeros(m1) = zeros(1,size(m1,2))
Zygote.@nograd getzeros


function inducing_kldiv(mm::SFGP)

    gp = mm.gp
    
    m1 = gp.m
    S1 = transpose(gp.L)*gp.L
    
    m2 = getzeros(m1)
    S2 = kernelmatrix(gp.k,gp.I)
    
    return kldiv(m1,S1,m2,S2)
end



function elbo(mm::SFGP,x,y)
    
    loglike = lpdf(mm,x,y)
    
    kld = inducing_kldiv(mm)
    
    return -loglike+kld

end


function sample_elbo(mm::SFGP, x, y, mc_size = 100)
     
    N = size(x,2)
    mcs = min(N,mc_size)
    
    samp = rand(1:N,mcs)
    
    xs = x[:,samp]
    ys = y[:,samp]
    
    loglike = lpdf(mm,xs,ys)/mcs
    kld = inducing_kldiv(mm)
    
    return -loglike+kld/N


end




function observed(mm::SFGP,m,s)
    probs = getProbs(m,s,mm.lower,mm.upper)[:]
    vals = (mm.vals)[:]
    
    dist =  Bernoulli(vals,probs)
    
    mm = mean(dist)
    ss = std(dist)
    lower = quantile(dist,0.05)
    upper = quantile(dist,0.95)
    
    return mm,ss,lower,upper,dist
end
